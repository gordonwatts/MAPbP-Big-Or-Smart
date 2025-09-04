# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and the data data25_13p6TeV:data25_13p6TeV.00505142.physics_Main.deriv.DAOD_PHYSLITE.f1621_m2272_p6828 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Uses ServiceX (FuncADL, PHYSLITE) to read exactly the requested datasets (one file each; NFiles=1 to keep it quick). If you want more statistics, increase NFiles yourself later.
- Converts all jet pT from MeV to GeV at the ServiceX level.
- Computes the leading-jet pT per event with Awkward (argmax pattern; no Python loops/lists).
- Builds a stacked MC histogram (JZ2 + JZ3) and overlays data as black points, restricted to 100–200 GeV.
- Applies event weights for MC: per-event mcEventWeight(0) scaled by cross section and integrated luminosity of the data in this request. Cross sections used:
  - 801167 (JZ2): 2.5826e9 pb
  - 801168 (JZ3): 2.8528e7 pb
- Estimates the data luminosity from the number of data events returned vs. the data25 benchmark (7.5e8 events = 39.3 fb^-1), and scales MC to this L. The plot is annotated with the L used.
- Writes the plot to leading_jet_pt_stacked.png.

```python
import awkward as ak
import numpy as np
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ATLAS)

# Dataset DIDs (copied exactly as requested)
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_data = "data25_13p6TeV:data25_13p6TeV.00505142.physics_Main.deriv.DAOD_PHYSLITE.f1621_m2272_p6828"

# Base query against PHYSLITE
base = FuncADLQueryPHYSLITE()

# Query for MC: per-event jets (pt in GeV) and the first MC event weight
mc_query = base.Select(lambda e: {
    "jet_pt": e.Jets().Select(lambda j: j.pt() / 1000.0),  # GeV
    "mcw": e.EventInfo("EventInfo").mcEventWeight(0),
})

# Query for data: per-event jets (pt in GeV)
data_query = base.Select(lambda e: {
    "jet_pt": e.Jets().Select(lambda j: j.pt() / 1000.0),  # GeV
})

# Deliver all in one shot (1 file per dataset for a quick example run)
spec = ServiceXSpec(
    Sample=[
        Sample(Name="JZ2", Dataset=dataset.Rucio(ds_jz2), NFiles=1, Query=mc_query),
        Sample(Name="JZ3", Dataset=dataset.Rucio(ds_jz3), NFiles=1, Query=mc_query),
        Sample(Name="data25", Dataset=dataset.Rucio(ds_data), NFiles=1, Query=data_query),
    ]
)

result = to_awk(deliver(spec))

# Helper: leading jet pT per event using argmax pattern (handles empty events -> None)
def leading_pt_per_event(jet_pts_by_event: ak.Array) -> ak.Array:
    # jet_pts_by_event is jagged: events * var * float
    argmax_idx = ak.argmax(jet_pts_by_event, axis=1, keepdims=True)
    lead = ak.firsts(jet_pts_by_event[argmax_idx])  # one value or None per event
    return lead

# Extract per-sample arrays
ev_jz2 = result["JZ2"]
ev_jz3 = result["JZ3"]
ev_data = result["data25"]

# Compute leading-jet pT per event
lead_jz2 = leading_pt_per_event(ev_jz2.jet_pt)
lead_jz3 = leading_pt_per_event(ev_jz3.jet_pt)
lead_data = leading_pt_per_event(ev_data.jet_pt)

# Data luminosity estimate:
# data25_13p6TeV has 7.5e8 events total -> 39.3 fb^-1 (benchmark).
# Rescale by the fraction of events we actually processed.
n_data_events = len(ev_data)
L_total_fb = 39.3
N_total_data25_events = 750_000_000
L_data_fb = L_total_fb * (n_data_events / N_total_data25_events if N_total_data25_events > 0 else 0.0)

# Cross sections in pico-barn for the MC samples (from provided table)
xsec_pb = {
    "JZ2": 2.5826e9,   # 801167
    "JZ3": 2.8528e7,   # 801168
}

# Sum of MC event weights (before any cuts)
sumw_jz2 = ak.sum(ev_jz2.mcw) if len(ev_jz2) > 0 else 0.0
sumw_jz3 = ak.sum(ev_jz3.mcw) if len(ev_jz3) > 0 else 0.0

# Scale factors: sf = L * sigma / N_S
# Convert sigma from pb to fb by dividing by 1000 (since L is in fb^-1)
sf_jz2 = (L_data_fb * (xsec_pb["JZ2"] / 1000.0)) / sumw_jz2 if sumw_jz2 != 0 else 0.0
sf_jz3 = (L_data_fb * (xsec_pb["JZ3"] / 1000.0)) / sumw_jz3 if sumw_jz3 != 0 else 0.0

# Event-level selections for histogram range: 100–200 GeV
def select_range(vals: ak.Array, low: float, high: float) -> ak.Array:
    # Remove None's, then apply range
    non_null_mask = ~ak.is_none(vals)
    vals_nn = vals[non_null_mask]
    in_range = (vals_nn >= low) & (vals_nn <= high)
    return vals_nn[in_range], non_null_mask, in_range

# Prepare MC fills (apply per-event mc weight, then sample-level scale factor)
lead_jz2_in, nn_mask_jz2, in_range_jz2 = select_range(lead_jz2, 100.0, 200.0)
lead_jz3_in, nn_mask_jz3, in_range_jz3 = select_range(lead_jz3, 100.0, 200.0)
lead_data_in, _, _ = select_range(lead_data, 100.0, 200.0)

mcw_jz2_in = ev_jz2.mcw[nn_mask_jz2][in_range_jz2] if len(ev_jz2) > 0 else ak.Array([])
mcw_jz3_in = ev_jz3.mcw[nn_mask_jz3][in_range_jz3] if len(ev_jz3) > 0 else ak.Array([])

w_jz2 = mcw_jz2_in * sf_jz2
w_jz3 = mcw_jz3_in * sf_jz3

# Build histograms
# MC histogram with categorical process axis, weighted
h_mc = (
    Hist.new.Reg(50, 100.0, 200.0, name="pt", label="Leading jet $p_{T}$ [GeV]")
        .StrCat(["JZ2", "JZ3"], name="process", label="Process")
        .Weight()
)
h_mc.fill(pt=lead_jz2_in, process="JZ2", weight=w_jz2)
h_mc.fill(pt=lead_jz3_in, process="JZ3", weight=w_jz3)

# Data histogram as counts (unweighted)
h_data = Hist.new.Reg(50, 100.0, 200.0, name="pt", label="Leading jet $p_{T}$ [GeV]").Int64()
h_data.fill(pt=lead_data_in)

# Plot
fig, ax = plt.subplots()

# Stacked MC
h_mc.plot(ax=ax, histtype="fill", stack=True, label="MC")

# Overlay data as black points with error bars
h_data.plot(ax=ax, histtype="errorbar", color="black", label="Data")

ax.set_xlim(100.0, 200.0)
ax.set_title("Leading jet $p_{T}$")
ax.legend()

# Annotate with the integrated luminosity used for MC scaling
ax.text(
    0.98, 0.95,
    f"L = {L_data_fb:.3f} fb$^{{-1}}$",
    transform=ax.transAxes,
    ha="right",
    va="top"
)

fig.tight_layout()
fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)

# Notes for the user printed to console (optional; not required for plot)
print("Applied weights:")
print(f"  - Data: weight = 1")
print(f"  - MC JZ2: event weight = mcEventWeight(0) * sf,  sf = {sf_jz2:.3e},  xsec = {xsec_pb['JZ2']} pb")
print(f"  - MC JZ3: event weight = mcEventWeight(0) * sf,  sf = {sf_jz3:.3e},  xsec = {xsec_pb['JZ3']} pb")
print(f"Data luminosity estimate used for scaling MC: L = {L_data_fb:.6f} fb^-1 (from {n_data_events} data events)")
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:30
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:30
data25: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
         Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
Applied weights:
  - Data: weight = 1
  - MC JZ2: event weight = mcEventWeight(0) * sf,  sf = 1.294e+03,  xsec = 2582600000.0 pb
  - MC JZ3: event weight = mcEventWeight(0) * sf,  sf = 3.593e+02,  xsec = 28528000.0 pb
Data luminosity estimate used for scaling MC: L = 0.005549 fb^-1 (from 105896 data events)
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
4f4fb700ef54: Waiting
660b8664e6e1: Waiting
ce8f13a8c0b6: Waiting
eb522b4f7276: Waiting
8f2cf14ad14d: Waiting
2a1478e0415a: Waiting
235451d01566: Waiting
a0831920243c: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
8fdd7df8e834: Verifying Checksum
8fdd7df8e834: Download complete
a0831920243c: Verifying Checksum
a0831920243c: Download complete
2a1478e0415a: Download complete
4f4fb700ef54: Verifying Checksum
235451d01566: Verifying Checksum
235451d01566: Download complete
ce8f13a8c0b6: Verifying Checksum
ce8f13a8c0b6: Download complete
eb522b4f7276: Download complete
660b8664e6e1: Verifying Checksum
660b8664e6e1: Download complete
59e22667830b: Pull complete
0f40319da806: Pull complete
8f2cf14ad14d: Verifying Checksum
8f2cf14ad14d: Download complete
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
/work/.venv/lib/python3.13/site-packages/mplhep/utils.py:741: UserWarning: Integer weights indicate poissonian data. Will calculate Garwood interval if ``scipy`` is installed. Otherwise errors will be set to ``sqrt(w2)``.
  self.errors()

```


</div></details>


![03a40394_gpt-5_leading_jet_pt_stacked.png](img/03a40394_gpt-5_leading_jet_pt_stacked.png)
<details><summary>Usage</summary>
<div style="margin-left: 1em;">


### LLM Usage
| Step | Time (sec) | Prompt Tokens | Completion Tokens | Total Tokens | Cost (USD) |
|---|---|---|---|---|---|
| Run 1 | 235.51 | 15109 | 8450 | 23559 | $0.10 |
| **Total** | 235.51 | 15109 | 8450 | 23559 | $0.103 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 52.94 |
| **Total** | **52.94** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,235.51,15109,8450,23559,0.103,1,52.94,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 235.51 | 15109 | 8450 | 23559 | $0.103 | 1 | 52.94 | Success |
