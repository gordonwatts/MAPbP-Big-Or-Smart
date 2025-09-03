# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Uses ServiceX (func_adl, PHYSLITE) to fetch only what’s needed from the two requested GRID datasets, filtering jets at the ServiceX level to 100 < pT < 200 GeV to minimize data transfer.
- Computes, per event, the leading-jet pT (if any jet is in the 100–200 GeV window) with Awkward only, no Python loops.
- Applies MC event weights and scales each MC sample to an integrated luminosity of 1 fb^-1 using cross sections:
  - 801167 (JZ2): 2,582,600,000.0 pb
  - 801168 (JZ3): 28,528,000.0 pb
  The scale factor per sample is sf = (L_pb × sigma_pb) / N_total, with L_pb = 1000 × L_fb and N_total counted before any cuts (i.e., all events).
- Builds a weighted, stacked histogram (100–200 GeV, 50 bins) of the leading jet pT and saves the plot to leading_jet_pt_stacked.png.
- Notes:
  - NFiles=1 is used for each dataset; when you are ready you can increase this locally.
  - You can adjust the luminosity by changing lumifb in the code.

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

# Datasets (as requested)
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Cross sections (pb), from provided table
xsec_pb = {
    "JZ2": 2582600000.0,  # 801167
    "JZ3": 28528000.0,    # 801168
}

# Target integrated luminosity (fb^-1)
lumifb = 1.0  # change if desired
L_pb = lumifb * 1000.0  # convert fb^-1 to pb^-1

# Build a single query to fetch:
# - per-event MC weight
# - a per-event constant "one" to count total events before cuts
# - per-event jets with pT in (100, 200) GeV (to reduce data moved)
base_query = FuncADLQueryPHYSLITE()
query = (
    base_query
    .Select(
        lambda e: {
            "evt_weight": e.EventInfo("EventInfo").mcEventWeight(0),
            "one": 1.0,
            "jet_pt": e.Jets()
                       .Where(lambda j: (j.pt() / 1000.0 > 100.0) and (j.pt() / 1000.0 < 200.0))
                       .Select(lambda j: j.pt() / 1000.0),
        }
    )
)

# Deliver both datasets in one call
all_data = to_awk(
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

data_jz2 = all_data["JZ2"]
data_jz3 = all_data["JZ3"]

def leading_pt_and_weights(sample_data, sample_name):
    # Total events BEFORE any cuts
    n_total = len(sample_data["one"])

    # Compute per-sample scale factor (MC xsec scaling to the chosen luminosity)
    sigma = xsec_pb[sample_name]
    sf = (L_pb * sigma) / n_total

    # Leading jet pT (per event) within 100–200 GeV (already filtered at ServiceX)
    # Sort each event's jet list descending and take the first
    sorted_pts = ak.sort(sample_data["jet_pt"], axis=1, ascending=False)
    leading_pt = ak.firsts(sorted_pts)

    # Keep only events that have a leading jet in this window (i.e., not None)
    good = ~ak.is_none(leading_pt)
    leading_pt = leading_pt[good]

    # Event weights: MC event weight times sample scale factor
    evt_w = sample_data["evt_weight"][good] * sf

    return leading_pt, evt_w

lp_jz2, w_jz2 = leading_pt_and_weights(data_jz2, "JZ2")
lp_jz3, w_jz3 = leading_pt_and_weights(data_jz3, "JZ3")

# Create a weighted histogram with a process category axis for stacking
h = (
    Hist.new
        .Reg(50, 100.0, 200.0, name="jet_pt", label="Leading jet $p_{T}$ [GeV]")
        .StrCat([], name="process", label="Process", growth=True)
        .Weight()
)

# Fill histogram for each process
h.fill(jet_pt=ak.to_numpy(lp_jz2), process="JZ2", weight=ak.to_numpy(w_jz2))
h.fill(jet_pt=ak.to_numpy(lp_jz3), process="JZ3", weight=ak.to_numpy(w_jz3))

# Plot stacked
fig, ax = plt.subplots()
# Plot each slice stacked on the same axes
h[:, "JZ2"].plot(ax=ax, histtype="fill", linewidth=1, edgecolor="gray", label="JZ2", stack=True)
h[:, "JZ3"].plot(ax=ax, histtype="fill", linewidth=1, edgecolor="gray", label="JZ3", stack=True)

ax.legend()
ax.set_xlabel("Leading jet $p_{T}$ [GeV]")
ax.set_ylabel("Events")
ax.set_title("Leading jet $p_{T}$ (stacked)")

fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)

print("Wrote: leading_jet_pt_stacked.png")
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:30
Wrote: leading_jet_pt_stacked.png
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
2a1478e0415a: Waiting
235451d01566: Waiting
a0831920243c: Waiting
4f4fb700ef54: Waiting
ce8f13a8c0b6: Waiting
8f2cf14ad14d: Waiting
660b8664e6e1: Waiting
eb522b4f7276: Waiting
8fdd7df8e834: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
2a1478e0415a: Verifying Checksum
2a1478e0415a: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
a0831920243c: Download complete
4f4fb700ef54: Verifying Checksum
4f4fb700ef54: Download complete
8fdd7df8e834: Verifying Checksum
8fdd7df8e834: Download complete
235451d01566: Verifying Checksum
235451d01566: Download complete
ce8f13a8c0b6: Verifying Checksum
ce8f13a8c0b6: Download complete
59e22667830b: Pull complete
660b8664e6e1: Verifying Checksum
660b8664e6e1: Download complete
eb522b4f7276: Verifying Checksum
eb522b4f7276: Download complete
0f40319da806: Pull complete
081f96d6584f: Pull complete
8fdd7df8e834: Pull complete
2a1478e0415a: Pull complete
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
| Run 1 | 94.45 | 14751 | 7004 | 21755 | $0.09 |
| **Total** | 94.45 | 14751 | 7004 | 21755 | $0.088 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 48.12 |
| **Total** | **48.12** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,94.45,14751,7004,21755,0.088,1,48.12,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 94.45 | 14751 | 7004 | 21755 | $0.088 | 1 | 48.12 | Success |
