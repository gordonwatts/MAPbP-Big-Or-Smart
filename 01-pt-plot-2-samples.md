# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Uses ServiceX (FuncADL, PHYSLITE) to fetch only the per-event jet pT values needed, filtered at source to jets with pT > 100 GeV to minimize I/O.
- Extracts the leading-jet pT per event with Awkward Array (no Python loops/lists), and applies the MC event weight (EventInfo.mcEventWeight(0)).
- Produces a stacked histogram (JZ2 + JZ3) of leading-jet pT between 100 and 200 GeV with 50 bins using Hist; plotted with ATLAS style and saved to a PNG file.
- Only NFiles=1 is fetched from each dataset as a quick, test-sized run; increase locally when ready.
- Cross-section scaling is NOT applied because a cross section value for 801168 (JZ3) was not available in the provided table; per instructions, when any sample’s cross section is missing, do not apply cross-section scaling to any samples. The plot is MC-event-weighted only.

```python
import awkward as ak
import numpy as np
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Style for plotting
plt.style.use(hep.style.ATLAS)

# Dataset names (copied exactly as requested)
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Build a single query that:
# - returns per-event list of jet pT in GeV for jets with pT > 100 GeV
# - returns the MC event weight
base_query = FuncADLQueryPHYSLITE()
query = base_query.Select(
    lambda e: {
        "jet_pt": e.Jets()
                   .Where(lambda j: j.pt() / 1000.0 > 100.0)
                   .Select(lambda j: j.pt() / 1000.0),
        "mc_w": e.EventInfo("EventInfo").mcEventWeight(0),
    }
)

# Deliver both datasets in a single ServiceX call (NFiles=1 each)
result = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="JZ2",
                    Dataset=dataset.Rucio(ds_jz2),
                    NFiles=1,
                    Query=query,
                ),
                Sample(
                    Name="JZ3",
                    Dataset=dataset.Rucio(ds_jz3),
                    NFiles=1,
                    Query=query,
                ),
            ]
        )
    )
)

data_jz2 = result["JZ2"]
data_jz3 = result["JZ3"]

# Helper to compute leading-jet pT per event and select 100–200 GeV
def leading_pt_in_range(data):
    jet_pt = data["jet_pt"]  # jagged array of per-event jet pT's [GeV], filtered >100 GeV
    w = data["mc_w"]         # per-event MC weight (scalar)

    # Find index of the max pT jet per event; keepdims=True to slice correctly
    max_idx = ak.argmax(jet_pt, axis=1, keepdims=True)
    lead_pt = ak.firsts(jet_pt[max_idx])  # optional values (None if no jets passed the 100 GeV cut)

    # Drop events with no selected jets (None) and keep only 100–200 GeV range
    valid = ~ak.is_none(lead_pt)
    lead_pt = lead_pt[valid]
    w = w[valid]

    # Histogram range selection
    in_range = (lead_pt >= 100.0) & (lead_pt <= 200.0)
    lead_pt = lead_pt[in_range]
    w = w[in_range]

    return lead_pt, w

lead_pt_jz2, w_jz2 = leading_pt_in_range(data_jz2)
lead_pt_jz3, w_jz3 = leading_pt_in_range(data_jz3)

# Build a single histogram with a process category axis
h_all = (
    Hist.new
        .Reg(50, 100.0, 200.0, name="pt", label="Leading jet $p_{T}$ [GeV]")
        .StrCat([], name="process", label="Process", growth=True)
        .Weight()
)

# Fill histogram for each process; convert Awkward -> NumPy for Hist.fill
h_all.fill(pt=ak.to_numpy(lead_pt_jz2), process="JZ2", weight=ak.to_numpy(w_jz2))
h_all.fill(pt=ak.to_numpy(lead_pt_jz3), process="JZ3", weight=ak.to_numpy(w_jz3))

# Prepare stacked plot using mplhep.histplot and Hist binning
fig, ax = plt.subplots()

# Extract process slices and plot stacked
h_jz2 = h_all[:, "JZ2"]
h_jz3 = h_all[:, "JZ3"]

# Values and edges for mplhep.histplot
values = [h_jz2.values(), h_jz3.values()]
edges = h_all.axes["pt"].edges

import mplhep as hep
hep.histplot(values, edges, stack=True, histtype="fill", label=["JZ2", "JZ3"], ax=ax)

ax.set_xlabel("Leading jet $p_{T}$ [GeV]")
ax.set_ylabel("Events (MC-weighted)")
ax.set_xlim(100, 200)
ax.set_title("Leading Jet $p_{T}$")
ax.legend()

fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)

print("Wrote plot to leading_jet_pt_stacked.png")
print("Note: Applied MC event weights only; no cross-section scaling applied (missing cross section for run 801168).")
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:26
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:30
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:21
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
Wrote plot to leading_jet_pt_stacked.png
Note: Applied MC event weights only; no cross-section scaling applied (missing cross section for run 801168).
**Success**

```

### stderr:

```text
Unable to find image 'gordonwatts/atlasplotagent:1.0.0' locally
1.0.0: Pulling from gordonwatts/atlasplotagent
59e22667830b: Pulling fs layer
0f40319da806: Pulling fs layer
081f96d6584f: Pulling fs layer
8fdd7df8e834: Pulling fs layer
2a1478e0415a: Pulling fs layer
235451d01566: Pulling fs layer
a0831920243c: Pulling fs layer
4f4fb700ef54: Pulling fs layer
ce8f13a8c0b6: Pulling fs layer
8f2cf14ad14d: Pulling fs layer
660b8664e6e1: Pulling fs layer
eb522b4f7276: Pulling fs layer
4f4fb700ef54: Waiting
ce8f13a8c0b6: Waiting
8fdd7df8e834: Waiting
2a1478e0415a: Waiting
8f2cf14ad14d: Waiting
660b8664e6e1: Waiting
235451d01566: Waiting
eb522b4f7276: Waiting
a0831920243c: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
8fdd7df8e834: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
2a1478e0415a: Download complete
a0831920243c: Verifying Checksum
a0831920243c: Download complete
4f4fb700ef54: Download complete
59e22667830b: Pull complete
ce8f13a8c0b6: Download complete
0f40319da806: Pull complete
660b8664e6e1: Verifying Checksum
660b8664e6e1: Download complete
081f96d6584f: Pull complete
8fdd7df8e834: Pull complete
2a1478e0415a: Pull complete
eb522b4f7276: Verifying Checksum
eb522b4f7276: Download complete
235451d01566: Verifying Checksum
235451d01566: Download complete
235451d01566: Pull complete
a0831920243c: Pull complete
4f4fb700ef54: Pull complete
ce8f13a8c0b6: Pull complete
8f2cf14ad14d: Verifying Checksum
8f2cf14ad14d: Download complete
8f2cf14ad14d: Pull complete
660b8664e6e1: Pull complete
eb522b4f7276: Pull complete
Digest: sha256:50780f1eaa7c0986e2e395adf88e50acb6e449689144926f8bd0942460f8d20a
Status: Downloaded newer image for gordonwatts/atlasplotagent:1.0.0
bash: cannot set terminal process group (-1): Inappropriate ioctl for device
bash: no job control in this shell
/work/.venv/lib/python3.13/site-packages/func_adl_servicex_xaodr25/trigger.py:127: SyntaxWarning: invalid escape sequence '\D'
  object. Close match is done as a function of $\Delta R < `dr`$.

```


</div></details>


![e1255027_gpt-5_leading_jet_pt_stacked.png](img/e1255027_gpt-5_leading_jet_pt_stacked.png)
<details><summary>Usage</summary>
<div style="margin-left: 1em;">


### LLM Usage
| Step | Time (sec) | Prompt Tokens | Completion Tokens | Total Tokens | Cost (USD) |
|---|---|---|---|---|---|
| Run 1 | 142.57 | 14634 | 5624 | 20258 | $0.07 |
| **Total** | 142.57 | 14634 | 5624 | 20258 | $0.075 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 51.87 |
| **Total** | **51.87** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,142.57,14634,5624,20258,0.075,1,51.87,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 142.57 | 14634 | 5624 | 20258 | $0.075 | 1 | 51.87 | Success |
