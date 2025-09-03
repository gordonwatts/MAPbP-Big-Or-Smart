# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Fetches both requested PHYSLITE datasets via one ServiceX deliver call (NFiles=1 for each sample), returning only what’s needed: per-event jet pT (in GeV) and the event MC weight.
- Computes the leading-jet pT per event with Awkward (no Python loops), then selects events with 100 ≤ pT < 200 GeV.
- Applies MC event weights and cross-section scaling for each dataset, scaled to L = 1 fb^-1:
  - JZ2 (801167): σ = 2.5826e9 pb
  - JZ3 (801168): σ = 2.8528e7 pb
- Builds a weighted, stacked histogram (process categories = JZ2, JZ3) of the leading-jet pT between 100 and 200 GeV using Hist and plots it with mplhep’s ATLAS style.
- Saves the plot to leading_jet_pt_stacked.png.
- Note: Only one file per dataset is processed (NFiles=1). Increase NFiles yourself when you are ready.

```python
import awkward as ak
from servicex import deliver, ServiceXSpec, Sample, dataset
from servicex_analysis_utils import to_awk
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE

from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Datasets (Rucio DIDs)
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Physics and plotting parameters
pt_low, pt_high = 100.0, 200.0  # GeV
n_bins = 50
L_fb = 1.0  # integrated luminosity used for MC scaling, in fb^-1

# Known cross sections (pb)
xsec_pb = {
    "JZ2": 2_582_600_000.0,  # 801167
    "JZ3":    28_528_000.0,  # 801168
}

# ----------------------------------------------------------------------
# Build one query to use for both datasets
#  - Return per-event list of jet pts (in GeV)
#  - Return per-event MC event weight
#  - Filter jets at ServiceX level to pt > 100 GeV to minimize egress
# ----------------------------------------------------------------------
base = FuncADLQueryPHYSLITE()
query = (
    base
    .Select(lambda e: {
        "jets": e.Jets(),  # collection
        "w": e.EventInfo("EventInfo").mcEventWeight(0)  # scalar per event
    })
    .Select(lambda d: {
        "jet_pt": d.jets
                     .Where(lambda j: j.pt()/1000.0 > 100.0)  # only jets with pt > 100 GeV
                     .Select(lambda j: j.pt()/1000.0),        # convert to GeV
        "event_weight": d.w
    })
)

# ----------------------------------------------------------------------
# Fetch both datasets in a single deliver call (NFiles=1 each)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Helper to compute leading-jet pt per event and scaled event weights
# ----------------------------------------------------------------------
def leading_pt_and_scaled_weights(per_event_pts, per_event_w, label):
    # Sum of MC event weights over all events (before any cuts)
    sumw = ak.sum(per_event_w)

    # Convert cross section from pb to fb to match L in fb^-1
    sigma_fb = xsec_pb[label] * 1e3
    # Scale factor for the entire sample
    sf = L_fb * sigma_fb / float(sumw)

    # Find index of max-pt jet per event
    # Returns [[idx], [None], ...] with keepdims=True to align for slicing
    max_idx = ak.argmax(per_event_pts, axis=1, keepdims=True)

    # Select the leading pt per event (None where no jets >100 GeV were found)
    leading_pt = per_event_pts[max_idx]
    leading_pt = ak.flatten(leading_pt)  # 1D, may include None's

    # Build mask: valid leading jet and within requested pT range [100, 200)
    valid_mask = ~ak.is_none(leading_pt)
    leading_pt = leading_pt[valid_mask]
    w = per_event_w[valid_mask] * sf

    in_range = (leading_pt >= pt_low) & (leading_pt < pt_high)
    return leading_pt[in_range], w[in_range]

# Compute leading-pt and scaled weights for JZ2 and JZ3
jz2_pt, jz2_w = leading_pt_and_scaled_weights(data_jz2["jet_pt"], data_jz2["event_weight"], "JZ2")
jz3_pt, jz3_w = leading_pt_and_scaled_weights(data_jz3["jet_pt"], data_jz3["event_weight"], "JZ3")

# ----------------------------------------------------------------------
# Build a stacked histogram with categorical axis for process
# ----------------------------------------------------------------------
h = (
    Hist.new
        .Reg(n_bins, pt_low, pt_high, name="pt", label="Leading jet $p_{T}$ [GeV]")
        .StrCat(["JZ2", "JZ3"], name="process", label="Process")
        .Weight()
)

# Fill per-process
h.fill(pt=jz2_pt, process="JZ2", weight=jz2_w)
h.fill(pt=jz3_pt, process="JZ3", weight=jz3_w)

# ----------------------------------------------------------------------
# Plot and save
# ----------------------------------------------------------------------
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
# Stacked filled histogram by the categorical axis
h.plot(stack=True, histtype="fill", ax=ax)

ax.set_xlim(pt_low, pt_high)
ax.set_xlabel("Leading jet $p_{T}$ [GeV]")
ax.set_title("Leading jet $p_{T}$ (100–200 GeV)")
ax.legend(title="Samples")

# Annotate the luminosity and note on weights
ax.text(
    0.02, 0.95,
    f"ATLAS Internal\nL = {L_fb:.1f} fb$^{{-1}}$\nMC weights + cross-section scaling",
    transform=ax.transAxes,
    va="top",
)

fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
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
8f2cf14ad14d: Waiting
660b8664e6e1: Waiting
a0831920243c: Waiting
eb522b4f7276: Waiting
4f4fb700ef54: Waiting
2a1478e0415a: Waiting
8fdd7df8e834: Waiting
ce8f13a8c0b6: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
8fdd7df8e834: Download complete
2a1478e0415a: Download complete
a0831920243c: Verifying Checksum
a0831920243c: Download complete
4f4fb700ef54: Verifying Checksum
4f4fb700ef54: Download complete
ce8f13a8c0b6: Verifying Checksum
ce8f13a8c0b6: Download complete
59e22667830b: Pull complete
0f40319da806: Pull complete
660b8664e6e1: Verifying Checksum
660b8664e6e1: Download complete
eb522b4f7276: Verifying Checksum
eb522b4f7276: Download complete
081f96d6584f: Pull complete
8fdd7df8e834: Pull complete
2a1478e0415a: Pull complete
235451d01566: Verifying Checksum
235451d01566: Download complete
8f2cf14ad14d: Verifying Checksum
8f2cf14ad14d: Download complete
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
| Run 1 | 85.47 | 14722 | 4833 | 19555 | $0.07 |
| **Total** | 85.47 | 14722 | 4833 | 19555 | $0.067 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 45.62 |
| **Total** | **45.62** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,85.47,14722,4833,19555,0.067,1,45.62,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 85.47 | 14722 | 4833 | 19555 | $0.067 | 1 | 45.62 | Success |
