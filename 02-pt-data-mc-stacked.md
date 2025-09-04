# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and the data data25_13p6TeV:data25_13p6TeV.00505142.physics_Main.deriv.DAOD_PHYSLITE.f1621_m2272_p6828 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Uses ServiceX (FuncADL R25, PHYSLITE) to fetch only the needed info from the three requested DIDs (1 file per sample as required).
- Extracts per-event jet pT (in GeV) and MC event weight (for MC samples) via ServiceX; leading-jet pT is computed with Awkward Array (no Python lists).
- Builds a stacked MC histogram (JZ2 + JZ3) of the leading jet pT between 100 and 200 GeV and overlays the data as black points.
- Applies MC event weights and cross-section scaling, normalized to the approximate integrated luminosity of the data slice processed:
  - Cross sections (from the provided table): JZ2 (run 801167) = 2,582,600,000.0 pb; JZ3 (run 801168) = 28,528,000.0 pb.
  - Data luminosity estimate for data25 is scaled by the number of events processed in this file: L_data ≈ 39.3 fb^-1 × N_events/data25_total_events (1.5e9).
- Produces and saves the plot to PNG. Note: This code intentionally uses NFiles=1 for each dataset; increase when you are ready.

```python
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Dataset IDs (copy exact)
DS_JZ2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
DS_JZ3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
DS_DATA25 = "data25_13p6TeV:data25_13p6TeV.00505142.physics_Main.deriv.DAOD_PHYSLITE.f1621_m2272_p6828"

# Build queries
# MC: per-event list of jet pt (GeV) and the MC event weight
q_mc = (
    FuncADLQueryPHYSLITE()
    .Select(lambda e: {
        "jet_pt": e.Jets().Select(lambda j: j.pt() / 1000.0),
        "mcw": e.EventInfo("EventInfo").mcEventWeight(0),
    })
)

# Data: per-event list of jet pt (GeV)
q_data = (
    FuncADLQueryPHYSLITE()
    .Select(lambda e: {
        "jet_pt": e.Jets().Select(lambda j: j.pt() / 1000.0),
    })
)

# ServiceX delivery (always NFiles=1 as requested)
spec = ServiceXSpec(
    Sample=[
        Sample(Name="mc_jz2", Dataset=dataset.Rucio(DS_JZ2), NFiles=1, Query=q_mc),
        Sample(Name="mc_jz3", Dataset=dataset.Rucio(DS_JZ3), NFiles=1, Query=q_mc),
        Sample(Name="data25", Dataset=dataset.Rucio(DS_DATA25), NFiles=1, Query=q_data),
    ]
)

all_data = to_awk(deliver(spec))

# Access returned awkward arrays
mc_jz2 = all_data["mc_jz2"]
mc_jz3 = all_data["mc_jz3"]
data25 = all_data["data25"]

# Helper to compute leading jet pT and per-event weights (Awkward-only)
def leading_pt_and_weights(jet_pts, evt_weights=None, pt_min=100.0, pt_max=200.0):
    # jet_pts: jagged array of per-event jet pt [GeV]
    have_jets = ak.num(jet_pts, axis=1) > 0
    jets_sel = jet_pts[have_jets]
    # argmax per event to get index of leading jet
    lead_idx = ak.argmax(jets_sel, axis=1, keepdims=True)
    lead_pt = ak.flatten(jets_sel[lead_idx], axis=1)  # 1D array
    # Apply the 100-200 GeV selection
    in_window = (lead_pt >= pt_min) & (lead_pt < pt_max)
    lead_pt = lead_pt[in_window]
    if evt_weights is not None:
        evt_w_sel = evt_weights[have_jets]
        evt_w_sel = evt_w_sel[in_window]
        return lead_pt, evt_w_sel
    else:
        return lead_pt, None

# Extract arrays
jz2_pts = mc_jz2["jet_pt"]
jz2_mcw = mc_jz2["mcw"]

jz3_pts = mc_jz3["jet_pt"]
jz3_mcw = mc_jz3["mcw"]

data_pts = data25["jet_pt"]

# Compute leading jet pT and select 100-200 GeV
pt_min, pt_max = 100.0, 200.0
jz2_lead_pt, jz2_evt_w = leading_pt_and_weights(jz2_pts, jz2_mcw, pt_min, pt_max)
jz3_lead_pt, jz3_evt_w = leading_pt_and_weights(jz3_pts, jz3_mcw, pt_min, pt_max)
data_lead_pt, _ = leading_pt_and_weights(data_pts, None, pt_min, pt_max)

# Compute total pre-cut sum of MC weights for normalization
# (sum over all events before any cuts)
jz2_sumw_all = ak.sum(jz2_mcw)
jz3_sumw_all = ak.sum(jz3_mcw)

# Estimate data luminosity for the processed sample (per provided guidance)
# data25 total: 1.5e9 events ~ 39.3 fb^-1
DATA25_TOTAL_EVENTS = 1_500_000_000
DATA25_TOTAL_LFB = 39.3
n_data_events = len(data_pts)  # number of events processed in this file
L_data_fb = DATA25_TOTAL_LFB * (n_data_events / DATA25_TOTAL_EVENTS)

# Cross sections from provided table (units: pb)
sigma_jz2_pb = 2_582_600_000.0  # 801167
sigma_jz3_pb = 28_528_000.0     # 801168

# Scale factors: sf = L (fb^-1) * 1000 pb/fb * sigma (pb) / sumw
def mc_scale_factor(L_fb, sigma_pb, sumw):
    if sumw == 0:
        return 0.0
    return L_fb * 1000.0 * sigma_pb / float(sumw)

sf_jz2 = mc_scale_factor(L_data_fb, sigma_jz2_pb, jz2_sumw_all)
sf_jz3 = mc_scale_factor(L_data_fb, sigma_jz3_pb, jz3_sumw_all)

# Build histograms
h_mc = (
    Hist.new
        .Reg(50, pt_min, pt_max, name="pt", label=r"Leading jet $p_T$ [GeV]")
        .StrCat([], name="process", label="Process", growth=True)
        .Weight()
)

# Fill MC with per-event weights times scale factors
if len(jz2_lead_pt) > 0:
    h_mc.fill(pt=jz2_lead_pt, process="JZ2", weight=jz2_evt_w * sf_jz2)
if len(jz3_lead_pt) > 0:
    h_mc.fill(pt=jz3_lead_pt, process="JZ3", weight=jz3_evt_w * sf_jz3)

# Data histogram (counts)
h_data = (
    Hist.new
        .Reg(50, pt_min, pt_max, name="pt", label=r"Leading jet $p_T$ [GeV]")
        .Weight()
)
if len(data_lead_pt) > 0:
    h_data.fill(pt=data_lead_pt, weight=ak.ones_like(data_lead_pt, dtype=float))

# Plot
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()

# Stacked MC
# (Hist will use the categorical axis to draw a stacked fill plot)
if h_mc.values().sum() > 0:
    h_mc.plot(stack=True, histtype="fill", linewidth=1, edgecolor="gray", ax=ax)

# Data overlay as points with error bars
if h_data.values().sum() > 0:
    h_data.plot(histtype="errorbar", color="black", label="Data", ax=ax)

ax.set_title("Leading jet $p_T$")
ax.legend()

# Annotate the (approximate) data luminosity used for MC normalization
ax.text(
    0.97, 0.95,
    f"L = {L_data_fb:.3e} fb$^{{-1}}$",
    ha="right", va="top", transform=ax.transAxes
)

fig.savefig("leading_jet_pt_100_200_stacked.png")
plt.close(fig)

print("Saved plot to leading_jet_pt_100_200_stacked.png")
print(f"Data luminosity used for MC normalization (approximate): L = {L_data_fb:.6e} fb^-1")
print("MC scaling applied: event weights and cross section normalization to the above L.")
```
### stdout:

```text
mc_jz2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:31
         Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:35
mc_jz3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
         Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
data25: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:26
         Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:30
Saved plot to leading_jet_pt_100_200_stacked.png
Data luminosity used for MC normalization (approximate): L = 2.774475e-03 fb^-1
MC scaling applied: event weights and cross section normalization to the above L.
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
8f2cf14ad14d: Waiting
660b8664e6e1: Waiting
2a1478e0415a: Waiting
235451d01566: Waiting
a0831920243c: Waiting
8fdd7df8e834: Waiting
eb522b4f7276: Waiting
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
/work/.venv/lib/python3.13/site-packages/mplhep/utils.py:741: UserWarning: Integer weights indicate poissonian data. Will calculate Garwood interval if ``scipy`` is installed. Otherwise errors will be set to ``sqrt(w2)``.
  self.errors()

```


</div></details>


![03a40394_gpt-5_leading_jet_pt_100_200_stacked.png](img/03a40394_gpt-5_leading_jet_pt_100_200_stacked.png)
<details><summary>Usage</summary>
<div style="margin-left: 1em;">


### LLM Usage
| Step | Time (sec) | Prompt Tokens | Completion Tokens | Total Tokens | Cost (USD) |
|---|---|---|---|---|---|
| Run 1 | 160.30 | 15110 | 7510 | 22620 | $0.09 |
| **Total** | 160.30 | 15110 | 7510 | 22620 | $0.094 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 53.26 |
| **Total** | **53.26** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,160.30,15110,7510,22620,0.094,1,53.26,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 160.30 | 15110 | 7510 | 22620 | $0.094 | 1 | 53.26 | Success |
