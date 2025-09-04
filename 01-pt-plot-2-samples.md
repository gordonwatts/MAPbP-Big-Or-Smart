# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Uses ServiceX (FuncADL, PHYSLITE, R25) to fetch only what’s needed from both Rucio datasets in a single delivery: per-event mcEventWeight(0) and jet pT (in GeV) for jets with pT > 100 GeV to reduce I/O.
- Computes the leading-jet pT per event with Awkward (argmax trick), keeps only events where the true leading-jet pT is between 100 and 200 GeV.
- Applies MC event weights and cross-section scaling to a target luminosity of L = 1 fb^-1:
  - JZ2 (run 801167): σ = 2,582,600,000.0 pb
  - JZ3 (run 801168): σ = 28,528,000.0 pb
  - Converts pb → fb in code and uses sum of per-event weights (before any cuts) for normalization.
- Builds a stacked 1D histogram (50 bins, 100–200 GeV) with Hist and plots with mplhep ATLAS style; saves to PNG.
- Runs on NFiles=1 for each dataset (recommended for quick tests); increase locally when ready.

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

# Dataset DIDs (copy-pasted exactly)
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Target luminosity for scaling (fb^-1)
target_L_fb = 1.0

# Cross sections (provided), in pb
xs_pb = {
    "JZ2": 2_582_600_000.0,  # pb
    "JZ3": 28_528_000.0,     # pb
}

# Build the query:
# - For each event: mcEventWeight(0)
# - Jets with pt > 100 GeV (to reduce data volume); convert to GeV on server side
base = FuncADLQueryPHYSLITE()
query = (
    base
    .Select(lambda e: {
        "jets": e.Jets(),
        "mcw": e.EventInfo("EventInfo").mcEventWeight(0),
    })
    .Select(lambda o: {
        "jet_pt": o.jets
            .Where(lambda j: j.pt() / 1000.0 > 100.0)
            .Select(lambda j: j.pt() / 1000.0),
        "mcw": o.mcw,
    })
)

# Deliver once for both datasets; NFiles=1 per guidance
samples = [
    Sample(Name="JZ2", Dataset=dataset.Rucio(ds_jz2), NFiles=1, Query=query),
    Sample(Name="JZ3", Dataset=dataset.Rucio(ds_jz3), NFiles=1, Query=query),
]

data_dict = to_awk(
    deliver(
        ServiceXSpec(Sample=samples),
    )
)

# Helper to compute leading-jet pT in [100, 200] GeV and aligned weights
def select_leading_pt_and_weights(sample_arr):
    # Per-event list of jet pt (GeV) for jets with pt>100 GeV
    jets_pt = sample_arr["jet_pt"]
    # Per-event MC weight
    w = sample_arr["mcw"]

    # Sum of weights BEFORE any selection (for normalization)
    sum_w_all = ak.sum(w)

    # Events with at least one jet >=100 GeV (as delivered)
    has_jets = ak.num(jets_pt, axis=1) > 0

    # Argmax trick to get the leading pT in each event
    idx = ak.argmax(jets_pt, axis=1, keepdims=True)
    leading_pt_all = ak.firsts(jets_pt[idx])  # may be None for events without jets

    # Keep only events where leading-jet exists and is within [100, 200] GeV
    # Note: jets_pt already ensures >=100 GeV; enforce upper bound here.
    valid_mask = (~ak.is_none(leading_pt_all)) & (leading_pt_all <= 200.0)
    leading_pt = leading_pt_all[valid_mask]

    # Align weights with selected events:
    # Start with events that had any jets, then apply the extra cut
    w_has_jet = w[has_jets]
    # valid_mask corresponds only to events with jets, so apply it to w_has_jet
    w_selected = w_has_jet[valid_mask]

    return leading_pt, w_selected, float(sum_w_all)

# Extract and prepare both samples
leading_pt_jz2, w_jz2, sumw_jz2 = select_leading_pt_and_weights(data_dict["JZ2"])
leading_pt_jz3, w_jz3, sumw_jz3 = select_leading_pt_and_weights(data_dict["JZ3"])

# Compute scale factors: sf = L * sigma / sum_w
# Convert pb -> fb (1 pb = 1000 fb)
sf_jz2 = target_L_fb * (xs_pb["JZ2"] * 1000.0) / sumw_jz2 if sumw_jz2 != 0.0 else 0.0
sf_jz3 = target_L_fb * (xs_pb["JZ3"] * 1000.0) / sumw_jz3 if sumw_jz3 != 0.0 else 0.0

# Build histogram: 50 bins, 100–200 GeV, with a categorical axis for process
hist_leadpt = (
    Hist.new
        .Reg(50, 100.0, 200.0, name="pt", label=r"Leading jet $p_{T}$ [GeV]")
        .StrCat([], name="process", label="Process", growth=True)
        .Weight()
)

# Fill stacked processes with appropriate weights (including scale factors)
hist_leadpt.fill(pt=leading_pt_jz2, process="JZ2", weight=w_jz2 * sf_jz2)
hist_leadpt.fill(pt=leading_pt_jz3, process="JZ3", weight=w_jz3 * sf_jz3)

# Plot (stacked)
fig, ax = plt.subplots()
hist_leadpt.plot(histtype="fill", stack=True, ax=ax)
ax.set_xlim(100, 200)
ax.set_title("Leading jet $p_{T}$")
ax.legend(title="Processes")
# Put only the target luminosity on the plot per guidance
ax.text(
    0.98, 0.95, f"L = {target_L_fb:.1f} fb$^{{-1}}$",
    transform=ax.transAxes, ha="right", va="top"
)

fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)

# Notes for the user (printed so you can verify scaling if desired)
print("Normalization details:")
print(f"  JZ2: xs = {xs_pb['JZ2']} pb, sumW = {sumw_jz2:.6g}, scale factor = {sf_jz2:.6g}")
print(f"  JZ3: xs = {xs_pb['JZ3']} pb, sumW = {sumw_jz3:.6g}, scale factor = {sf_jz3:.6g}")
print("Saved: leading_jet_pt_stacked.png")
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
ce8f13a8c0b6: Waiting
8f2cf14ad14d: Waiting
8fdd7df8e834: Waiting
660b8664e6e1: Waiting
eb522b4f7276: Waiting
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
2a1478e0415a: Verifying Checksum
a0831920243c: Verifying Checksum
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
Traceback (most recent call last):
  File "/app/script.py", line 87, in <module>
    leading_pt_jz2, w_jz2, sumw_jz2 = select_leading_pt_and_weights(data_dict["JZ2"])
                                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/app/script.py", line 82, in select_leading_pt_and_weights
    w_selected = w_has_jet[valid_mask]
                 ~~~~~~~~~^^^^^^^^^^^^
  File "/work/.venv/lib/python3.13/site-packages/awkward/highlevel.py", line 1104, in __getitem__
    with ak._errors.SlicingErrorContext(self, where):
         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/work/.venv/lib/python3.13/site-packages/awkward/_errors.py", line 80, in __exit__
    raise self.decorate_exception(exception_type, exception_value)
  File "/work/.venv/lib/python3.13/site-packages/awkward/highlevel.py", line 1112, in __getitem__
    indexed_layout = prepare_layout(self._layout._getitem(where, NamedAxis))
                                    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/work/.venv/lib/python3.13/site-packages/awkward/contents/content.py", line 657, in _getitem
    return self._getitem(where.layout, named_axis)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/.venv/lib/python3.13/site-packages/awkward/contents/content.py", line 734, in _getitem
    return self._getitem((where,), named_axis)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/work/.venv/lib/python3.13/site-packages/awkward/contents/content.py", line 649, in _getitem
    out = next._getitem_next(nextwhere[0], nextwhere[1:], None)
  File "/work/.venv/lib/python3.13/site-packages/awkward/contents/regulararray.py", line 732, in _getitem_next
    return self._getitem_next_missing(head, tail, advanced)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/work/.venv/lib/python3.13/site-packages/awkward/contents/content.py", line 490, in _getitem_next_missing
    nextcontent = self._getitem_next(headcontent, tail, advanced)
  File "/work/.venv/lib/python3.13/site-packages/awkward/contents/regulararray.py", line 596, in _getitem_next
    self._maybe_index_error(
    ~~~~~~~~~~~~~~~~~~~~~~~^
        self._backend[
        ^^^^^^^^^^^^^^
    ...<9 lines>...
        slicer=head,
        ^^^^^^^^^^^^
    )
    ^
  File "/work/.venv/lib/python3.13/site-packages/awkward/contents/content.py", line 295, in _maybe_index_error
    raise ak._errors.index_error(self, slicer, message)
IndexError: cannot slice RegularArray (of length 1) with [    1     3     6 ... 29995 29997 29999]: index out of range while attempting to get index 12653 (in compiled code: https://github.com/scikit-hep/awkward/blob/awkward-cpp-48/awkward-cpp/src/cpu-kernels/awkward_RegularArray_getitem_next_array_regularize.cpp#L19)

This error occurred while attempting to slice

    <Array [3.98e-05, 0.000107, ..., 1.92e-05, 0.000109] type='12653 * float32'>

with

    <Array [None, True, None, True, ..., True, None, True] type='30000 * ?bool'>

```


</div></details>
<details><summary>Run 2</summary>
<div style="margin-left: 1em;">


- Root cause: Misaligned boolean masking. The code created valid_mask over all events (length = number of events), but applied it to w_has_jet, which had already been filtered to only events with at least one jet. Indexing an array of length N_has with a mask of length N led to the Awkward IndexError.
- Secondary issue: Using argmax across all events included empty-jet events, producing None entries that then complicated masking. While handled via ak.firsts, combining masks from different event selections caused the shape mismatch.
- Fix: First select only events with ≥1 jet (has_jets) and apply all subsequent computations (leading jet pT and the [100, 200] GeV cut) and weight alignment on this subset. This guarantees all masks and arrays are aligned in size and structure. No change to the ServiceX query (still filtering jets at >100 GeV on the server), and plotting/output conventions are preserved.

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

# Dataset DIDs (copy-pasted exactly)
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Target luminosity for scaling (fb^-1)
target_L_fb = 1.0

# Cross sections (provided), in pb
xs_pb = {
    "JZ2": 2_582_600_000.0,  # pb
    "JZ3": 28_528_000.0,     # pb
}

# Build the query:
# - For each event: mcEventWeight(0)
# - Jets with pt > 100 GeV (to reduce data volume); convert to GeV on server side
base = FuncADLQueryPHYSLITE()
query = (
    base
    .Select(lambda e: {
        "jets": e.Jets(),
        "mcw": e.EventInfo("EventInfo").mcEventWeight(0),
    })
    .Select(lambda o: {
        "jet_pt": o.jets
            .Where(lambda j: j.pt() / 1000.0 > 100.0)
            .Select(lambda j: j.pt() / 1000.0),
        "mcw": o.mcw,
    })
)

# Deliver once for both datasets; NFiles=1 per guidance
samples = [
    Sample(Name="JZ2", Dataset=dataset.Rucio(ds_jz2), NFiles=1, Query=query),
    Sample(Name="JZ3", Dataset=dataset.Rucio(ds_jz3), NFiles=1, Query=query),
]

data_dict = to_awk(
    deliver(
        ServiceXSpec(Sample=samples),
    )
)

# Helper to compute leading-jet pT in [100, 200] GeV and aligned weights
def select_leading_pt_and_weights(sample_arr):
    # Per-event list of jet pt (GeV) for jets with pt>100 GeV
    jets_pt = sample_arr["jet_pt"]
    # Per-event MC weight
    w = sample_arr["mcw"]

    # Sum of weights BEFORE any selection (for normalization)
    sum_w_all = ak.sum(w)

    # Select only events with at least one jet >=100 GeV (as delivered)
    has_jets = ak.num(jets_pt, axis=1) > 0
    jets_pt_has = jets_pt[has_jets]
    w_has = w[has_jets]

    # Leading jet pT for events with jets
    idx = ak.argmax(jets_pt_has, axis=1, keepdims=True)
    leading_pt_has = ak.firsts(jets_pt_has[idx])

    # Apply the upper bound cut (jets are already >=100 GeV from the query)
    keep = leading_pt_has <= 200.0
    leading_pt = leading_pt_has[keep]
    w_selected = w_has[keep]

    return leading_pt, w_selected, float(sum_w_all)

# Extract and prepare both samples
leading_pt_jz2, w_jz2, sumw_jz2 = select_leading_pt_and_weights(data_dict["JZ2"])
leading_pt_jz3, w_jz3, sumw_jz3 = select_leading_pt_and_weights(data_dict["JZ3"])

# Compute scale factors: sf = L * sigma / sum_w
# Convert pb -> fb (1 pb = 1000 fb)
sf_jz2 = target_L_fb * (xs_pb["JZ2"] * 1000.0) / sumw_jz2 if sumw_jz2 != 0.0 else 0.0
sf_jz3 = target_L_fb * (xs_pb["JZ3"] * 1000.0) / sumw_jz3 if sumw_jz3 != 0.0 else 0.0

# Build histogram: 50 bins, 100–200 GeV, with a categorical axis for process
hist_leadpt = (
    Hist.new
        .Reg(50, 100.0, 200.0, name="pt", label=r"Leading jet $p_{T}$ [GeV]")
        .StrCat([], name="process", label="Process", growth=True)
        .Weight()
)

# Fill stacked processes with appropriate weights (including scale factors)
hist_leadpt.fill(pt=leading_pt_jz2, process="JZ2", weight=w_jz2 * sf_jz2)
hist_leadpt.fill(pt=leading_pt_jz3, process="JZ3", weight=w_jz3 * sf_jz3)

# Plot (stacked)
fig, ax = plt.subplots()
hist_leadpt.plot(histtype="fill", stack=True, ax=ax)
ax.set_xlim(100, 200)
ax.set_title("Leading jet $p_{T}$")
ax.legend(title="Processes")
# Put only the target luminosity on the plot per guidance
ax.text(
    0.98, 0.95, f"L = {target_L_fb:.1f} fb$^{{-1}}$",
    transform=ax.transAxes, ha="right", va="top"
)

fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)

# Notes for the user (printed so you can verify scaling if desired)
print("Normalization details:")
print(f"  JZ2: xs = {xs_pb['JZ2']} pb, sumW = {sumw_jz2:.6g}, scale factor = {sf_jz2:.6g}")
print(f"  JZ3: xs = {xs_pb['JZ3']} pb, sumW = {sumw_jz3:.6g}, scale factor = {sf_jz3:.6g}")
print("Saved: leading_jet_pt_stacked.png")
```
### stdout:

```text

Normalization details:
  JZ2: xs = 2582600000.0 pb, sumW = 11.078, scale factor = 2.33128e+11
  JZ3: xs = 28528000.0 pb, sumW = 0.440641, scale factor = 6.4742e+10
Saved: leading_jet_pt_stacked.png
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
| Run 1 | 189.87 | 15068 | 7033 | 22101 | $0.09 |
| Run 2 | 97.94 | 18174 | 3311 | 21485 | $0.06 |
| **Total** | 287.81 | 33242 | 10344 | 43586 | $0.145 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 47.75 |
| Run 2 | 7.08 |
| **Total** | **54.83** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,287.81,33242,10344,43586,0.145,2,54.83,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 287.81 | 33242 | 10344 | 43586 | $0.145 | 2 | 54.83 | Success |
