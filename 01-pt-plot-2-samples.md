# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Uses ServiceX (func_adl_servicex_xaodr25) to fetch only events that contain at least one jet with pT > 100 GeV from both requested DAOD_PHYSLITE datasets (NFiles=1 each, as recommended).
- Returns, per event, the list of jet pT (in GeV) for jets above 100 GeV and the MC event weight; then uses Awkward to pick the highest-pT jet per event.
- Builds a stacked 1D histogram of the leading-jet pT between 100 and 200 GeV with Hist, and saves it to a PNG file.
- Applies MC event weights. Cross-section scaling is prepared but not applied by default because the total number of events in each full sample (before any cuts) is not known here. Fill in total_events_in_sample[...] to enable L·σ/N scaling (L defaults to 1 fb^-1, σ values included for JZ2/JZ3).
- All momenta are converted to GeV as soon as they are read back; manipulation is performed with Awkward Arrays; plotting uses Hist + mplhep (ATLAS style).

```python
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Plot style
plt.style.use(hep.style.ATLAS)

# Dataset DIDs (copy exactly)
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Build a single query that:
# - keeps only events with at least one jet with pT > 100 GeV
# - returns the pT (in GeV) of jets above 100 GeV and the event MC weight
base = FuncADLQueryPHYSLITE()
query = (
    base
    .Where(
        lambda e: e.Jets()
                  .Where(lambda j: j.pt() / 1000.0 > 100.0)
                  .Count() > 0
    )
    .Select(
        lambda e: {
            "jet_pt": e.Jets()
                       .Where(lambda j: j.pt() / 1000.0 > 100.0)
                       .Select(lambda j: j.pt() / 1000.0),
            "mcw": e.EventInfo("EventInfo").mcEventWeight(0),
        }
    )
)

# Deliver both datasets in a single call (NFiles=1 as recommended)
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

jz2 = results["JZ2"]
jz3 = results["JZ3"]

# Extract leading-jet pT per event (Awkward-only ops)
def leading_pt_and_weights(sample):
    # jet_pt is an event-wise jagged list of pT>100 GeV jets (in GeV)
    jet_pts = sample["jet_pt"]
    # Find index of max pT per event; keepdims for correct jagged slicing
    max_idx = ak.argmax(jet_pts, axis=1, keepdims=True)
    leading_pt = ak.flatten(jet_pts[max_idx])
    # Event MC weight aligned with events that passed the event-level filter
    mcw = sample["mcw"]
    return leading_pt, mcw

jz2_lead_pt, jz2_mcw = leading_pt_and_weights(jz2)
jz3_lead_pt, jz3_mcw = leading_pt_and_weights(jz3)

# Optional cross-section scaling setup (L * sigma / N_total)
# - L in fb^-1
# - sigma in fb (converted below)
# - N_total should be the TOTAL number of events in the full sample BEFORE any cuts
L_fb = 1.0  # target luminosity (fb^-1)

# Cross sections from the provided table (convert to fb)
# 1 mb = 1e12 fb, 1 µb = 1e9 fb
sigma_fb = {
    "JZ2": 2.58e12,  # 2.58 mb -> 2.58e12 fb
    "JZ3": 8.6e9,    # 8.6 µb -> 8.6e9 fb
}

# Fill in the totals for the FULL samples if you want L·σ/N scaling applied.
# Leave as None to skip cross-section scaling (only MC event weights applied).
total_events_in_sample = {
    "JZ2": None,  # e.g., replace with the true total number of events for JZ2
    "JZ3": None,  # e.g., replace with the true total number of events for JZ3
}

# Compute per-sample scale factors; if N_total is unknown, skip cross-section scaling
scale = {}
for name in ["JZ2", "JZ3"]:
    N_total = total_events_in_sample[name]
    if N_total is None:
        scale[name] = 1.0
    else:
        scale[name] = L_fb * sigma_fb[name] / float(N_total)

# Apply weights (Awkward ops; casting to NumPy only when filling hist)
jz2_weights = jz2_mcw * scale["JZ2"]
jz3_weights = jz3_mcw * scale["JZ3"]

# Build a stacked histogram in the 100–200 GeV range
h = (
    Hist.new
        .Reg(50, 100.0, 200.0, name="pt", label=r"Leading jet $p_{T}$ [GeV]")
        .StrCat(["JZ2", "JZ3"], name="sample", label="Sample")
        .Weight()
)

# Fill histogram
h.fill(
    pt=ak.to_numpy(jz2_lead_pt),
    sample="JZ2",
    weight=ak.to_numpy(jz2_weights),
)
h.fill(
    pt=ak.to_numpy(jz3_lead_pt),
    sample="JZ3",
    weight=ak.to_numpy(jz3_weights),
)

# Plot stacked fill
fig, ax = plt.subplots()
h.plot(stack=True, histtype="fill", edgecolor="gray", linewidth=1, ax=ax)
ax.set_xlabel(r"Leading jet $p_{T}$ [GeV]")
ax.set_title(r"Leading jet $p_{T}$")
ax.legend(title="Sample")

fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)

# Informative message if cross-section scaling was skipped
if any(v is None for v in total_events_in_sample.values()):
    print("Note: Cross-section scaling (L·σ/N) was not applied because total event counts for the full samples were not provided. Only MC event weights were used.")
    print("To enable scaling, fill in total_events_in_sample[...] with the FULL pre-cut event counts for each dataset.")
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:30

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
8fdd7df8e834: Waiting
235451d01566: Waiting
2a1478e0415a: Waiting
a0831920243c: Waiting
ce8f13a8c0b6: Waiting
8f2cf14ad14d: Waiting
660b8664e6e1: Waiting
eb522b4f7276: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
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
660b8664e6e1: Verifying Checksum
660b8664e6e1: Download complete
235451d01566: Verifying Checksum
235451d01566: Download complete
eb522b4f7276: Verifying Checksum
eb522b4f7276: Download complete
59e22667830b: Pull complete
0f40319da806: Pull complete
081f96d6584f: Pull complete
8fdd7df8e834: Pull complete
2a1478e0415a: Pull complete
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
/work/.venv/lib/python3.13/site-packages/hist/quick_construct.py:381: UserWarning: sample is a protected keyword and cannot be used as axis name
  return self.hist_class(
Traceback (most recent call last):
  File "/app/script.py", line 110, in <module>
    h.fill(
    ~~~~~~^
        pt=ak.to_numpy(jz2_lead_pt),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        sample="JZ2",
        ^^^^^^^^^^^^^
        weight=ak.to_numpy(jz2_weights),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/work/.venv/lib/python3.13/site-packages/hist/basehist.py", line 259, in fill
    raise TypeError(
        "All axes must be accounted for in fill, you may have used a disallowed name in the axes"
    )
TypeError: All axes must be accounted for in fill, you may have used a disallowed name in the axes

```


</div></details>
<details><summary>Run 2</summary>
<div style="margin-left: 1em;">


- The categorical axis in the histogram was named "sample", which is a protected keyword in Hist. Hist warned about this and then failed the fill with “All axes must be accounted for…” because the disallowed axis name prevented proper matching.
- Fix: rename the categorical axis to a safe name (e.g., "dataset") and update the corresponding fill calls and legend title accordingly.
- Everything else (ServiceX query, Awkward manipulation, plotting to PNG) is fine and unchanged.

```python
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Plot style
plt.style.use(hep.style.ATLAS)

# Dataset DIDs (copy exactly)
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Build a single query that:
# - keeps only events with at least one jet with pT > 100 GeV
# - returns the pT (in GeV) of jets above 100 GeV and the event MC weight
base = FuncADLQueryPHYSLITE()
query = (
    base
    .Where(
        lambda e: e.Jets()
                  .Where(lambda j: j.pt() / 1000.0 > 100.0)
                  .Count() > 0
    )
    .Select(
        lambda e: {
            "jet_pt": e.Jets()
                       .Where(lambda j: j.pt() / 1000.0 > 100.0)
                       .Select(lambda j: j.pt() / 1000.0),
            "mcw": e.EventInfo("EventInfo").mcEventWeight(0),
        }
    )
)

# Deliver both datasets in a single call (NFiles=1 as recommended)
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

jz2 = results["JZ2"]
jz3 = results["JZ3"]

# Extract leading-jet pT per event (Awkward-only ops)
def leading_pt_and_weights(sample):
    # jet_pt is an event-wise jagged list of pT>100 GeV jets (in GeV)
    jet_pts = sample["jet_pt"]
    # Find index of max pT per event; keepdims for correct jagged slicing
    max_idx = ak.argmax(jet_pts, axis=1, keepdims=True)
    leading_pt = ak.flatten(jet_pts[max_idx])
    # Event MC weight aligned with events that passed the event-level filter
    mcw = sample["mcw"]
    return leading_pt, mcw

jz2_lead_pt, jz2_mcw = leading_pt_and_weights(jz2)
jz3_lead_pt, jz3_mcw = leading_pt_and_weights(jz3)

# Optional cross-section scaling setup (L * sigma / N_total)
# - L in fb^-1
# - sigma in fb (converted below)
# - N_total should be the TOTAL number of events in the full sample BEFORE any cuts
L_fb = 1.0  # target luminosity (fb^-1)

# Cross sections from the provided table (convert to fb)
# 1 mb = 1e12 fb, 1 µb = 1e9 fb
sigma_fb = {
    "JZ2": 2.58e12,  # 2.58 mb -> 2.58e12 fb
    "JZ3": 8.6e9,    # 8.6 µb -> 8.6e9 fb
}

# Fill in the totals for the FULL samples if you want L·σ/N scaling applied.
# Leave as None to skip cross-section scaling (only MC event weights applied).
total_events_in_sample = {
    "JZ2": None,  # e.g., replace with the true total number of events for JZ2
    "JZ3": None,  # e.g., replace with the true total number of events for JZ3
}

# Compute per-sample scale factors; if N_total is unknown, skip cross-section scaling
scale = {}
for name in ["JZ2", "JZ3"]:
    N_total = total_events_in_sample[name]
    if N_total is None:
        scale[name] = 1.0
    else:
        scale[name] = L_fb * sigma_fb[name] / float(N_total)

# Apply weights (Awkward ops; casting to NumPy only when filling hist)
jz2_weights = jz2_mcw * scale["JZ2"]
jz3_weights = jz3_mcw * scale["JZ3"]

# Build a stacked histogram in the 100–200 GeV range
# NOTE: "sample" is a protected keyword in Hist; use a different name like "dataset"
h = (
    Hist.new
        .Reg(50, 100.0, 200.0, name="pt", label=r"Leading jet $p_{T}$ [GeV]")
        .StrCat(["JZ2", "JZ3"], name="dataset", label="Dataset")
        .Weight()
)

# Fill histogram
h.fill(
    pt=ak.to_numpy(jz2_lead_pt),
    dataset="JZ2",
    weight=ak.to_numpy(jz2_weights),
)
h.fill(
    pt=ak.to_numpy(jz3_lead_pt),
    dataset="JZ3",
    weight=ak.to_numpy(jz3_weights),
)

# Plot stacked fill
fig, ax = plt.subplots()
h.plot(stack=True, histtype="fill", edgecolor="gray", linewidth=1, ax=ax)
ax.set_xlabel(r"Leading jet $p_{T}$ [GeV]")
ax.set_title(r"Leading jet $p_{T}$")
ax.legend(title="Dataset")

fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)

# Informative message if cross-section scaling was skipped
if any(v is None for v in total_events_in_sample.values()):
    print("Note: Cross-section scaling (L·σ/N) was not applied because total event counts for the full samples were not provided. Only MC event weights were used.")
    print("To enable scaling, fill in total_events_in_sample[...] with the FULL pre-cut event counts for each dataset.")
```
### stdout:

```text

Note: Cross-section scaling (L·σ/N) was not applied because total event counts for the full samples were not provided. Only MC event weights were used.
To enable scaling, fill in total_events_in_sample[...] with the FULL pre-cut event counts for each dataset.
**Success**

```

### stderr:

```text
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
| Run 1 | 163.15 | 14660 | 6564 | 21224 | $0.08 |
| Run 2 | 66.62 | 17096 | 2283 | 19379 | $0.04 |
| **Total** | 229.77 | 31756 | 8847 | 40603 | $0.128 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 47.85 |
| Run 2 | 6.90 |
| **Total** | **54.75** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,229.77,31756,8847,40603,0.128,2,54.75,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 229.77 | 31756 | 8847 | 40603 | $0.128 | 2 | 54.75 | Success |
