# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Uses ServiceX (FuncADL, PHYSLITE) to fetch only what’s needed from both requested Rucio datasets, with jets filtered at the ServiceX level to pT > 100 GeV (in GeV) to reduce data movement.
- Returns per-event: list of jet pT’s (GeV) and the MC event weight (mcEventWeight(0)); no event-level cuts are applied so the event-weight sum is valid for cross-section scaling.
- Computes the leading-jet pT per event with Awkward Array operations (no Python loops), keeps only events where a jet above 100 GeV exists, and applies per-sample weights:
  - MC event weights.
  - Cross-section scaling with L = 1 fb^-1 and σ from the provided table:
    - 801167 (JZ2): 2,582,600,000.0 pb
    - 801168 (JZ3): 28,528,000.0 pb
- Builds a Hist histogram with a categorical “process” axis and fills “JZ2” and “JZ3”; plots a stacked histogram between 100 and 200 GeV with 50 bins and saves to leading_jet_pt_stacked.png.
- Note: Only NFiles=1 is run for each dataset (recommended for iteration). Increase NFiles yourself when you are ready.

```python
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

plt.style.use(hep.style.ATLAS)

# Datasets (Rucio DIDs) exactly as requested
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Physics and scaling configuration
L_fb = 1.0  # target integrated luminosity in fb^-1 (change if desired)

# Cross sections from the provided table (pb)
xsec_pb = {
    "JZ2": 2_582_600_000.0,  # 801167
    "JZ3": 28_528_000.0,     # 801168
}

# Build the base query: per event, get jets (pT in GeV) filtered to pT>100 GeV and the MC event weight
base = FuncADLQueryPHYSLITE()
query = (
    base
    .Select(lambda e: {
        "jet_pt": e.Jets().Where(lambda j: j.pt() / 1000.0 > 100.0).Select(lambda j: j.pt() / 1000.0),
        "w": e.EventInfo("EventInfo").mcEventWeight(0),
    })
)

# Deliver both datasets in a single ServiceX call (use only NFiles=1 as per guidance)
results = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(Name="JZ2", Dataset=dataset.Rucio(ds_jz2), NFiles=1, Query=query),
                Sample(Name="JZ3", Dataset=dataset.Rucio(ds_jz3), NFiles=1, Query=query),
            ]
        )
    )
)

# Helper to compute leading-jet pT per event and corresponding event weights
def leading_pt_and_weights(arr):
    # arr has fields: jet_pt (jagged), w (per-event scalar)
    jet_pts = arr["jet_pt"]  # jagged lists of pT (already in GeV and >100 GeV)
    evt_w = arr["w"]         # per-event weight (float)

    # Leading jet pT per event: sort descending, take first; None if no jets survived the 100 GeV cut
    leading_pt_per_event = ak.firsts(ak.sort(jet_pts, axis=1, ascending=False))
    have_leading = ~ak.is_none(leading_pt_per_event)

    leading_pts = leading_pt_per_event[have_leading]
    weights = evt_w[have_leading]
    return leading_pts, weights, evt_w  # also return all event weights for normalization

# Extract for each sample
lead_jz2, w_selected_jz2, all_evt_w_jz2 = leading_pt_and_weights(results["JZ2"])
lead_jz3, w_selected_jz3, all_evt_w_jz3 = leading_pt_and_weights(results["JZ3"])

# Compute per-sample normalization factors:
# sf = L * sigma / sum_w, using sigma in fb (convert pb -> fb by *1000)
sumw_jz2 = float(ak.sum(all_evt_w_jz2))
sumw_jz3 = float(ak.sum(all_evt_w_jz3))

sigma_fb_jz2 = xsec_pb["JZ2"] * 1000.0
sigma_fb_jz3 = xsec_pb["JZ3"] * 1000.0

sf_jz2 = (L_fb * sigma_fb_jz2) / sumw_jz2 if sumw_jz2 != 0 else 0.0
sf_jz3 = (L_fb * sigma_fb_jz3) / sumw_jz3 if sumw_jz3 != 0 else 0.0

# Build histogram: 50 bins from 100 to 200 GeV, with a categorical "process" axis
hist = (
    Hist.new
        .Reg(50, 100.0, 200.0, name="pt", label="Leading jet $p_T$ [GeV]")
        .StrCat([], name="process", label="Process", growth=True)
        .Weight()
)

# Fill histogram for each process with weights (event weight * scaling factor)
hist.fill(
    pt=ak.to_numpy(lead_jz2),
    process="JZ2",
    weight=ak.to_numpy(w_selected_jz2 * sf_jz2),
)
hist.fill(
    pt=ak.to_numpy(lead_jz3),
    process="JZ3",
    weight=ak.to_numpy(w_selected_jz3 * sf_jz3),
)

# Plot stacked histogram
fig, ax = plt.subplots()
hist.plot(stack=True, histtype="fill", edgecolor="black", linewidth=0.5, ax=ax)
ax.set_xlim(100.0, 200.0)
ax.set_xlabel("Leading jet $p_T$ [GeV]")
ax.set_ylabel("Events")
ax.set_title("Leading jet $p_T$")
ax.legend(title="Process")

fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)

print("Wrote: leading_jet_pt_stacked.png")
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:21
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
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
8fdd7df8e834: Waiting
2a1478e0415a: Waiting
235451d01566: Waiting
a0831920243c: Waiting
4f4fb700ef54: Waiting
ce8f13a8c0b6: Waiting
8f2cf14ad14d: Waiting
660b8664e6e1: Waiting
eb522b4f7276: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
8fdd7df8e834: Verifying Checksum
8fdd7df8e834: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
2a1478e0415a: Download complete
a0831920243c: Download complete
4f4fb700ef54: Verifying Checksum
4f4fb700ef54: Download complete
ce8f13a8c0b6: Verifying Checksum
ce8f13a8c0b6: Download complete
235451d01566: Verifying Checksum
235451d01566: Download complete
660b8664e6e1: Verifying Checksum
660b8664e6e1: Download complete
eb522b4f7276: Verifying Checksum
eb522b4f7276: Download complete
59e22667830b: Pull complete
8f2cf14ad14d: Verifying Checksum
8f2cf14ad14d: Download complete
0f40319da806: Pull complete
081f96d6584f: Pull complete
8fdd7df8e834: Pull complete
2a1478e0415a: Pull complete
235451d01566: Pull complete
a0831920243c: Pull complete
4f4fb700ef54: Pull complete
ce8f13a8c0b6: Pull complete
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
| Run 1 | 129.13 | 14691 | 6250 | 20941 | $0.08 |
| **Total** | 129.13 | 14691 | 6250 | 20941 | $0.081 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 44.05 |
| **Total** | **44.05** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,129.13,14691,6250,20941,0.081,1,44.05,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 129.13 | 14691 | 6250 | 20941 | $0.081 | 1 | 44.05 | Success |
