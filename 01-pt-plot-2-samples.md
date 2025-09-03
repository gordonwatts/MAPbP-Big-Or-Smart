# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Fetches both rucio datasets with a single ServiceX request (NFiles=1 for each) and brings back only what’s needed: per-event jet pT (in GeV), the MC event weight, and a per-event counter for event counting.
- Uses Awkward Array to find the leading-jet pT per event via ak.argmax, selects events where the leading-jet pT is in [100, 200] GeV, and applies weights.
- Applies MC event weights and cross-section scaling to 1 fb^-1 luminosity:
  - JZ2 (run 801167): 2.58 milli-barn
  - JZ3 (run 801168): 8.6 micro-barn
- Builds a weighted histogram with a categorical “process” axis and produces a stacked plot saved to leading_jet_pt_100_200_stacked.png.
- Note: Only one file per dataset (NFiles=1) is processed here; increase NFiles yourself when you’re ready to run more.

```python
import awkward as ak
import numpy as np
import vector
vector.register_awkward()

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ATLAS)

# Dataset names (Rucio DIDs) – copied exactly as requested
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Base query for PHYSLITE
base_query = FuncADLQueryPHYSLITE()

# Query: per-event jets' pT (GeV), MC event weight, and an event counter
query = (
    base_query
    .Select(lambda e: {
        "jet_pt": e.Jets().Select(lambda j: j.pt() / 1000.0),  # GeV
        "mcw": e.EventInfo("EventInfo").mcEventWeight(0),
        "one": 1.0,
    })
)

# ServiceX delivery: both datasets in a single request; keep NFiles=1
spec = ServiceXSpec(
    Sample=[
        Sample(
            Name="JZ2",
            Dataset=dataset.Rucio(ds_jz2),
            NFiles=1,  # Keep to 1; scale yourself when ready
            Query=query,
        ),
        Sample(
            Name="JZ3",
            Dataset=dataset.Rucio(ds_jz3),
            NFiles=1,  # Keep to 1; scale yourself when ready
            Query=query,
        ),
    ]
)

all_data = to_awk(deliver(spec))
d_jz2 = all_data["JZ2"]
d_jz3 = all_data["JZ3"]

# Helper to compute leading-jet pT per event and weights, applying selection 100–200 GeV
def select_leading_pt_and_weights(d, sigma_fb):
    # Leading jet pT per event using argmax
    # Keepdims gives [[idx]] so we can slice; then ak.firsts to get a 1D array with Nones where needed
    lead_idx = ak.argmax(d["jet_pt"], axis=1, keepdims=True)
    lead_pt = ak.firsts(d["jet_pt"][lead_idx])

    # Event selection: leading jet pT in [100, 200] GeV and not None
    in_range = (~ak.is_none(lead_pt)) & (lead_pt >= 100.0) & (lead_pt <= 200.0)

    # Count all events processed in this sample (before any cuts) for scaling
    n_events = len(d["one"])
    # Luminosity (fb^-1)
    L_fb = 1.0

    # Scale factor per-event for this sample: L * sigma / N_S
    sf = (L_fb * sigma_fb) / float(n_events) if n_events > 0 else 0.0

    # Total event weight = mc event weight * sf
    w = d["mcw"] * sf

    # Apply selection
    sel_pt = lead_pt[in_range]
    sel_w = w[in_range]

    # Convert to numpy for Hist.fill
    return ak.to_numpy(sel_pt), ak.to_numpy(sel_w)

# Cross sections converted to femto-barn (fb)
# 1 barn = 1e15 fb
sigma_fb_jz2 = 2.58e-3 * 1.0e15   # 2.58 milli-barn
sigma_fb_jz3 = 8.6e-6  * 1.0e15   # 8.6 micro-barn

pt_jz2, w_jz2 = select_leading_pt_and_weights(d_jz2, sigma_fb_jz2)
pt_jz3, w_jz3 = select_leading_pt_and_weights(d_jz3, sigma_fb_jz3)

# Histogram: 50 bins between 100 and 200 GeV, with categorical process axis; weighted storage
h = (
    Hist.new
        .Reg(50, 100.0, 200.0, name="pt", label=r"Leading jet $p_{T}$ [GeV]")
        .StrCat(["JZ2", "JZ3"], name="process", label="Process")
        .Weight()
)

# Fill for each process
h.fill(pt=pt_jz2, process="JZ2", weight=w_jz2)
h.fill(pt=pt_jz3, process="JZ3", weight=w_jz3)

# Make a stacked plot
fig, ax = plt.subplots()
h_stack = h.stack("process")
# Plot as stacked filled histograms
h_stack.plot(stack=True, histtype="fill", linewidth=1, edgecolor="black")
ax.set_xlim(100, 200)
ax.set_xlabel(r"Leading jet $p_{T}$ [GeV]")
ax.set_ylabel("Events (weighted)")
ax.set_title("Leading jet $p_{T}$")
ax.legend(title="Process")

fig.savefig("leading_jet_pt_100_200_stacked.png")
plt.close(fig)
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:21
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:21
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:26
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
8fdd7df8e834: Waiting
eb522b4f7276: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
8fdd7df8e834: Verifying Checksum
8fdd7df8e834: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
2a1478e0415a: Verifying Checksum
2a1478e0415a: Download complete
a0831920243c: Verifying Checksum
a0831920243c: Download complete
4f4fb700ef54: Download complete
ce8f13a8c0b6: Verifying Checksum
ce8f13a8c0b6: Download complete
660b8664e6e1: Verifying Checksum
660b8664e6e1: Download complete
eb522b4f7276: Verifying Checksum
eb522b4f7276: Download complete
59e22667830b: Pull complete
8f2cf14ad14d: Verifying Checksum
8f2cf14ad14d: Download complete
0f40319da806: Pull complete
235451d01566: Verifying Checksum
235451d01566: Download complete
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


![e1255027_gpt-5_leading_jet_pt_100_200_stacked.png](img/e1255027_gpt-5_leading_jet_pt_100_200_stacked.png)
<details><summary>Usage</summary>
<div style="margin-left: 1em;">


### LLM Usage
| Step | Time (sec) | Prompt Tokens | Completion Tokens | Total Tokens | Cost (USD) |
|---|---|---|---|---|---|
| Run 1 | 105.48 | 14746 | 5698 | 20444 | $0.08 |
| **Total** | 105.48 | 14746 | 5698 | 20444 | $0.075 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 47.31 |
| **Total** | **47.31** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,105.48,14746,5698,20444,0.075,1,47.31,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 105.48 | 14746 | 5698 | 20444 | $0.075 | 1 | 47.31 | Success |
