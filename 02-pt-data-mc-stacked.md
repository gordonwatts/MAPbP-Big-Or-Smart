# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and the data data25_13p6TeV:data25_13p6TeV.00505142.physics_Main.deriv.DAOD_PHYSLITE.f1621_m2272_p6828 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Uses ServiceX (FuncADL on PHYSLITE) to fetch exactly what’s needed from all three datasets in a single deliver call (NFiles=1 each).
- For MC (JZ2 and JZ3): returns per-event list of jet pT (only jets with pT ≥ 100 GeV to reduce I/O) and the per-event MC event weight (mcEventWeight(0)) for proper normalization.
- For data: returns per-event list of jet pT (only jets with pT ≥ 100 GeV).
- In Awkward: computes the leading jet pT per event by sorting each event’s jet pT list and taking the first entry; then selects values in 100–200 GeV for histogramming.
- Computes the data luminosity scale using the provided guidance for data25_13p6TeV: L_data = 39.3 fb^-1 × (N_data_events / 150e12) based on the number of data events processed.
- Applies MC weighting: per-event weight = mcEventWeight × (L_data × σ / sum(mcEventWeight over all events)), using σ(JZ2)=2.5826e9 pb and σ(JZ3)=2.8528e7 pb (converted to fb).
- Builds a stacked histogram for MC and overlays data as black points; writes plot to leading_jet_pt_stacked.png. The plot indicates only the integrated luminosity used (L=... fb^-1).
- Note: Only 1 file per dataset is processed (NFiles=1) as a safe default—feel free to increase locally to get more statistics.

```python
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Dataset names (DIDs)
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_data = "data25_13p6TeV:data25_13p6TeV.00505142.physics_Main.deriv.DAOD_PHYSLITE.f1621_m2272_p6828"

# Build queries
base_mc = FuncADLQueryPHYSLITE()
mc_query = (
    base_mc
    .Select(lambda e: {
        # Keep only jets with pt >= 100 GeV to minimize transfer; leading computed later.
        "jets_pt": e.Jets().Where(lambda j: j.pt() / 1000.0 >= 100.0).Select(lambda j: j.pt() / 1000.0),
        # MC event weight for normalization (sum taken over ALL events; no event-level filter!)
        "mc_evt_w": e.EventInfo("EventInfo").mcEventWeight(0),
    })
)

base_data = FuncADLQueryPHYSLITE()
data_query = (
    base_data
    .Select(lambda e: {
        "jets_pt": e.Jets().Where(lambda j: j.pt() / 1000.0 >= 100.0).Select(lambda j: j.pt() / 1000.0),
    })
)

# Deliver from ServiceX; always use NFiles=1 per instructions
result = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="JZ2",
                    Dataset=dataset.Rucio(ds_jz2),
                    NFiles=1,
                    Query=mc_query,
                ),
                Sample(
                    Name="JZ3",
                    Dataset=dataset.Rucio(ds_jz3),
                    NFiles=1,
                    Query=mc_query,
                ),
                Sample(
                    Name="DATA",
                    Dataset=dataset.Rucio(ds_data),
                    NFiles=1,
                    Query=data_query,
                ),
            ]
        ),
    )
)

jz2 = result["JZ2"]
jz3 = result["JZ3"]
data_arr = result["DATA"]

# Helper: leading jet pT per event from a per-event list of jet pT's
def leading_pt_from_lists(jets_pt_list):
    # Sort each event's jets by pT descending, then take the first (None if empty)
    sorted_pts = ak.sort(jets_pt_list, axis=1, ascending=False)
    leading = ak.firsts(sorted_pts)
    return leading

# Compute leading-jet pT arrays
lead_jz2 = leading_pt_from_lists(jz2["jets_pt"])
lead_jz3 = leading_pt_from_lists(jz3["jets_pt"])
lead_data = leading_pt_from_lists(data_arr["jets_pt"])

# Select leading-jet pT in the desired range [100, 200) GeV
def select_range(arr, lo=100.0, hi=200.0):
    mask = (~ak.is_none(arr)) & (arr >= lo) & (arr < hi)
    return arr[mask], mask

lead_jz2_in, mask_jz2 = select_range(lead_jz2, 100.0, 200.0)
lead_jz3_in, mask_jz3 = select_range(lead_jz3, 100.0, 200.0)
lead_data_in, mask_data = select_range(lead_data, 100.0, 200.0)

# Compute data luminosity for data25_13p6TeV using guidance
# Total events in year and total luminosity (fb^-1) for data25_13p6TeV
DATA25_TOTAL_EVENTS = 150_000_000_000_000  # 1.5e14
DATA25_TOTAL_LUMI_FB = 39.3

n_data_events_processed = len(data_arr["jets_pt"])  # number of events looked at (no event-level selection applied)
L_data_fb = DATA25_TOTAL_LUMI_FB * (float(n_data_events_processed) / float(DATA25_TOTAL_EVENTS))  # fb^-1

# Cross sections
# Provided in table (units): JZ2 and JZ3 are in pico-barn; convert to fb
pb_to_fb = 1000.0
sigma_jz2_fb = 2_582_600_000.0 * pb_to_fb  # 2.5826e9 pb -> fb
sigma_jz3_fb = 28_528_000.0 * pb_to_fb     # 2.8528e7 pb -> fb

# Sum of MC weights over all events (before cuts)
wsum_jz2 = ak.sum(jz2["mc_evt_w"])
wsum_jz3 = ak.sum(jz3["mc_evt_w"])

# Per-event MC weights for events that pass the 100<=pT<200 selection
def mc_event_weights_for_selected(sample, mask_selected, wsum, sigma_fb, L_fb):
    # Scale factor: L * sigma / sum_w
    # Protect against division by zero
    scale = (L_fb * sigma_fb / wsum) if wsum != 0 else 0.0
    # Event weights for all events
    w_all = sample["mc_evt_w"] * scale
    # Apply same mask as for the selected leading-pt entries
    w_sel = w_all[mask_selected]
    return w_sel

w_jz2_sel = mc_event_weights_for_selected(jz2, mask_jz2, wsum_jz2, sigma_jz2_fb, L_data_fb)
w_jz3_sel = mc_event_weights_for_selected(jz3, mask_jz3, wsum_jz3, sigma_jz3_fb, L_data_fb)

# Convert inputs to NumPy for Hist filling
lead_jz2_np = ak.to_numpy(lead_jz2_in)
lead_jz3_np = ak.to_numpy(lead_jz3_in)
lead_data_np = ak.to_numpy(lead_data_in)
w_jz2_np = ak.to_numpy(w_jz2_sel)
w_jz3_np = ak.to_numpy(w_jz3_sel)

# Build histograms
# MC stacked histogram with process category
h_mc = (
    Hist.new
    .Reg(50, 100.0, 200.0, name="pt", label="Leading jet $p_{T}$ [GeV]")
    .StrCat([], name="process", label="Process", growth=True)
    .Weight()
)
h_mc.fill(pt=lead_jz2_np, process="JZ2", weight=w_jz2_np)
h_mc.fill(pt=lead_jz3_np, process="JZ3", weight=w_jz3_np)

# Data histogram (counts)
h_data = (
    Hist.new
    .Reg(50, 100.0, 200.0, name="pt", label="Leading jet $p_{T}$ [GeV]")
    .Int64()
)
h_data.fill(pt=lead_data_np)

# Plot
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()

# Stacked MC
h_mc.plot(ax=ax, histtype="fill", stack=True, edgecolor="gray", linewidth=1, label=["JZ2", "JZ3"])

# Data overlay as points with error bars
h_data.plot(ax=ax, histtype="errorbar", color="black", label="Data")

ax.set_title("Leading jet $p_{T}$")
ax.legend()

# Put only the luminosity on the plot
ax.text(
    0.98, 0.95, f"L = {L_data_fb:.3e} fb$^{{-1}}$",
    transform=ax.transAxes, ha="right", va="top"
)

fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)

# Notes for the user (printed for transparency; can be removed if undesired)
print("Notes:")
print(" - Processed NFiles=1 for each dataset via ServiceX.")
print(" - MC normalized with per-event mcEventWeight and cross sections:")
print(f"   JZ2 σ = 2.5826e9 pb, JZ3 σ = 2.8528e7 pb (converted to fb).")
print(f" - Data luminosity estimate: L_data = 39.3 fb^-1 * (N_data_events / 1.5e14).")
print(f"   N_data_events processed = {n_data_events_processed}, L_data ≈ {L_data_fb:.6e} fb^-1.")
print(" - Leading jet pT computed by sorting per-event jets and taking the highest pT;")
print("   histogram filled only for 100 ≤ pT < 200 GeV.")
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:26
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:30
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:21
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
DATA: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:21
       Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
Notes:
 - Processed NFiles=1 for each dataset via ServiceX.
 - MC normalized with per-event mcEventWeight and cross sections:
   JZ2 σ = 2.5826e9 pb, JZ3 σ = 2.8528e7 pb (converted to fb).
 - Data luminosity estimate: L_data = 39.3 fb^-1 * (N_data_events / 1.5e14).
   N_data_events processed = 105896, L_data ≈ 2.774475e-08 fb^-1.
 - Leading jet pT computed by sorting per-event jets and taking the highest pT;
   histogram filled only for 100 ≤ pT < 200 GeV.
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
eb522b4f7276: Waiting
660b8664e6e1: Waiting
8fdd7df8e834: Waiting
2a1478e0415a: Waiting
235451d01566: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
8fdd7df8e834: Verifying Checksum
8fdd7df8e834: Download complete
2a1478e0415a: Download complete
a0831920243c: Verifying Checksum
a0831920243c: Download complete
4f4fb700ef54: Verifying Checksum
4f4fb700ef54: Download complete
235451d01566: Verifying Checksum
235451d01566: Download complete
ce8f13a8c0b6: Download complete
660b8664e6e1: Verifying Checksum
660b8664e6e1: Download complete
eb522b4f7276: Verifying Checksum
eb522b4f7276: Download complete
59e22667830b: Pull complete
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
| Run 1 | 186.34 | 15115 | 9065 | 24180 | $0.11 |
| **Total** | 186.34 | 15115 | 9065 | 24180 | $0.110 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 50.35 |
| **Total** | **50.35** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,186.34,15115,9065,24180,0.110,1,50.35,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 186.34 | 15115 | 9065 | 24180 | $0.110 | 1 | 50.35 | Success |
