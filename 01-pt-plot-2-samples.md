# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Fetches both requested mc23 PHYSLITE datasets via a single ServiceX delivery (NFiles=1 for each; you can increase later when ready).
- Skims only what’s needed from ServiceX: per-event mcEventWeight(0) and jet pT>50 GeV (converted to GeV at the source).
- Uses Awkward to compute the leading-jet pT per event and applies the 100–200 GeV selection.
- Applies MC event weights and cross-section scaling to L=1.0 fb^-1 using the provided table:
  - 801167 (JZ2): 2,582,600,000.0 pb
  - 801168 (JZ3): 28,528,000.0 pb
- Builds a weighted, stacked histogram (Hist) of leading-jet pT and saves to PNG.

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

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Datasets (copy-pasted exactly as requested)
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Target luminosity in fb^-1
L_target_fb = 1.0

# Cross-sections from the provided table (pb)
xsec_pb = {
    "JZ2": 2_582_600_000.0,  # 801167
    "JZ3": 28_528_000.0,     # 801168
}

# Convert cross-sections to fb (1 pb = 1e3 fb)
xsec_fb = {k: v * 1.0e3 for k, v in xsec_pb.items()}

# ----------------------------------------------------------------------
# Build the ServiceX query: per-event mc weight and jet pt list (>50 GeV) in GeV
# ----------------------------------------------------------------------
base_query = FuncADLQueryPHYSLITE()

query = (
    base_query
    .Select(lambda e: {
        "mc_weight": e.EventInfo("EventInfo").mcEventWeight(0),
        # Filter jets at the source to reduce data volume (we only care about leading jet >100 GeV later)
        "jet_pt": e.Jets().Where(lambda j: j.pt() / 1000.0 > 50.0).Select(lambda j: j.pt() / 1000.0),
    })
)

# ----------------------------------------------------------------------
# Deliver both datasets in a single call (NFiles=1 for each)
# ----------------------------------------------------------------------
results = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="JZ2",
                    Dataset=dataset.Rucio(ds_jz2),
                    NFiles=1,  # Always 1 here; increase when you are ready
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

data_jz2 = results["JZ2"]
data_jz3 = results["JZ3"]

# ----------------------------------------------------------------------
# Awkward processing: leading-jet pT per event and selection 100–200 GeV
# ----------------------------------------------------------------------
def leading_pt_in_window(arr, pt_min=100.0, pt_max=200.0):
    jet_pts = arr["jet_pt"]       # per-event list of jet pT (GeV)
    mc_w = arr["mc_weight"]       # per-event MC weight

    # Find leading jet pT per event
    # - ak.argmax(..., keepdims=True) returns indices shaped for slicing, with None for empty lists
    lead_idx = ak.argmax(jet_pts, axis=1, keepdims=True)
    lead_pt = ak.firsts(jet_pts[lead_idx])  # None if no jets in event

    # Build selection mask for the requested pT window
    mask = (~ak.is_none(lead_pt)) & (lead_pt >= pt_min) & (lead_pt < pt_max)

    # Selected leading pT and corresponding original MC weights
    sel_pt = lead_pt[mask]
    sel_w = mc_w[mask]
    return sel_pt, sel_w, mc_w  # also return all mc weights to compute normalization sum

# Process both datasets
sel_pt_jz2, sel_w_jz2, mcw_all_jz2 = leading_pt_in_window(data_jz2, 100.0, 200.0)
sel_pt_jz3, sel_w_jz3, mcw_all_jz3 = leading_pt_in_window(data_jz3, 100.0, 200.0)

# ----------------------------------------------------------------------
# Compute normalization factors (sum over ALL events, before any cuts)
# sf = L * sigma / sum_w
# ----------------------------------------------------------------------
sumw_jz2 = float(ak.sum(mcw_all_jz2))
sumw_jz3 = float(ak.sum(mcw_all_jz3))

sf_jz2 = (L_target_fb * xsec_fb["JZ2"]) / sumw_jz2 if sumw_jz2 != 0.0 else 0.0
sf_jz3 = (L_target_fb * xsec_fb["JZ3"]) / sumw_jz3 if sumw_jz3 != 0.0 else 0.0

print(f"Normalization:")
print(f"  JZ2: sum_w(all events) = {sumw_jz2:.6g}, xsec = {xsec_pb['JZ2']} pb, scale factor = {sf_jz2:.6g}")
print(f"  JZ3: sum_w(all events) = {sumw_jz3:.6g}, xsec = {xsec_pb['JZ3']} pb, scale factor = {sf_jz3:.6g}")

# Apply scale factors to selected events' weights
w_jz2 = ak.to_numpy(sel_w_jz2 * sf_jz2)
w_jz3 = ak.to_numpy(sel_w_jz3 * sf_jz3)
pt_jz2 = ak.to_numpy(sel_pt_jz2)
pt_jz3 = ak.to_numpy(sel_pt_jz3)

# ----------------------------------------------------------------------
# Make a stacked histogram with Hist
# ----------------------------------------------------------------------
h = (
    Hist.new
    .Reg(50, 100.0, 200.0, name="pt", label="Leading jet $p_{T}$ [GeV]")
    .StrCat([], name="process", label="Process", growth=True)
    .Weight()
)

# Fill per process
h.fill(pt=pt_jz2, process="JZ2", weight=w_jz2)
h.fill(pt=pt_jz3, process="JZ3", weight=w_jz3)

# Prepare slices for plotting
h_jz2 = h[:, "JZ2"]
h_jz3 = h[:, "JZ3"]

# ----------------------------------------------------------------------
# Plot (stacked) and save
# ----------------------------------------------------------------------
fig, ax = plt.subplots()

# Use mplhep to draw histograms stacked; Hist provides edges and values
hep.histplot(
    [h_jz2, h_jz3],
    stack=True,
    histtype="fill",
    label=["JZ2", "JZ3"],
    ax=ax,
)

ax.legend()
ax.set_xlabel("Leading jet $p_{T}$ [GeV]")
ax.set_ylabel("Events")
ax.set_title("Leading jet $p_{T}$")
ax.text(0.02, 0.98, f"L={L_target_fb:.1f} fb$^{{-1}}$", transform=ax.transAxes, ha="left", va="top")

fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)

print("Saved plot to leading_jet_pt_stacked.png")
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:21
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
Normalization:
  JZ2: sum_w(all events) = 11.078, xsec = 2582600000.0 pb, scale factor = 2.33128e+11
  JZ3: sum_w(all events) = 0.440641, xsec = 28528000.0 pb, scale factor = 6.4742e+10
Saved plot to leading_jet_pt_stacked.png
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
ce8f13a8c0b6: Waiting
8f2cf14ad14d: Waiting
2a1478e0415a: Waiting
235451d01566: Waiting
4f4fb700ef54: Waiting
660b8664e6e1: Waiting
a0831920243c: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
8fdd7df8e834: Verifying Checksum
8fdd7df8e834: Download complete
2a1478e0415a: Verifying Checksum
2a1478e0415a: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
a0831920243c: Verifying Checksum
a0831920243c: Download complete
4f4fb700ef54: Verifying Checksum
4f4fb700ef54: Download complete
ce8f13a8c0b6: Verifying Checksum
ce8f13a8c0b6: Download complete
235451d01566: Verifying Checksum
235451d01566: Download complete
660b8664e6e1: Download complete
eb522b4f7276: Verifying Checksum
eb522b4f7276: Download complete
8f2cf14ad14d: Verifying Checksum
8f2cf14ad14d: Download complete
59e22667830b: Pull complete
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
| Run 1 | 158.81 | 15067 | 7172 | 22239 | $0.09 |
| **Total** | 158.81 | 15067 | 7172 | 22239 | $0.091 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 47.30 |
| **Total** | **47.30** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,158.81,15067,7172,22239,0.091,1,47.30,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 158.81 | 15067 | 7172 | 22239 | $0.091 | 1 | 47.30 | Success |
