# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Fetch jets from both requested DAOD_PHYSLITE datasets using ServiceX (Release 25), filtering at the ServiceX level to keep only jets with pT > 100 GeV, and converting MeV to GeV.
- Use Awkward Array to compute the leading-jet pT per event (argmax per event), and keep only values within 100–200 GeV for plotting.
- Build a 1D Hist with a categorical axis for dataset names and fill it with the leading-jet pT values; plot as a stacked filled histogram in ATLAS style.
- Save the plot to leading_jet_pt_stacked_100_200.png.
- Note: Only NFiles=1 is fetched per dataset for speed. When you are ready, increase NFiles yourself.

```python
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Use ATLAS/mplhep style for plotting
plt.style.use(hep.style.ATLAS)

# Dataset DIDs requested by the user (Rucio)
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Build a query on PHYSLITE:
# - Filter jets at ServiceX level: pt > 100 GeV (to reduce data volume)
# - Return per-event list of jet pT (in GeV)
base = FuncADLQueryPHYSLITE()
query = (
    base
    .Select(lambda e: e.Jets().Where(lambda j: j.pt() / 1000.0 > 100.0))
    .Select(lambda jets: {
        "jet_pt": jets.Select(lambda j: j.pt() / 1000.0),
    })
)

# Deliver both datasets in a single ServiceX request (NFiles=1 for each)
samples = [
    Sample(Name="JZ2", Dataset=dataset.Rucio(ds_jz2), NFiles=1, Query=query),
    Sample(Name="JZ3", Dataset=dataset.Rucio(ds_jz3), NFiles=1, Query=query),
]
result = to_awk(
    deliver(ServiceXSpec(Sample=samples))
)

# Helper: compute leading jet pT per event with Awkward only
def leading_jet_pt_per_event(jet_pt_per_event: ak.Array) -> ak.Array:
    # Keep only events with at least one jet
    mask = ak.num(jet_pt_per_event, axis=1) > 0
    jets_non_empty = jet_pt_per_event[mask]
    # Argmax per event (keepdims for proper jagged indexing), then flatten to 1D
    idx = ak.argmax(jets_non_empty, axis=1, keepdims=True)
    leading = ak.flatten(jets_non_empty[idx], axis=1)
    # Keep within requested plotting range [100, 200] GeV
    leading = leading[(leading >= 100.0) & (leading <= 200.0)]
    return leading

# Extract arrays from ServiceX output and compute leading pT
jz2_jets = result["JZ2"]["jet_pt"]
jz3_jets = result["JZ3"]["jet_pt"]

leading_jz2 = leading_jet_pt_per_event(jz2_jets)
leading_jz3 = leading_jet_pt_per_event(jz3_jets)

# Build a histogram with a dataset category axis and pt axis [100, 200] GeV
h = (
    Hist.new
    .Reg(50, 100.0, 200.0, name="pt", label="Leading jet $p_{T}$ [GeV]")
    .StrCat(["JZ2", "JZ3"], name="dataset", label="Dataset")
    .Int64()
)

# Fill histogram; convert to NumPy for Hist.fill
pt_jz2 = ak.to_numpy(leading_jz2)
pt_jz3 = ak.to_numpy(leading_jz3)

h.fill(pt=pt_jz2, dataset=np.full(pt_jz2.shape, "JZ2", dtype=object))
h.fill(pt=pt_jz3, dataset=np.full(pt_jz3.shape, "JZ3", dtype=object))

# Plot stacked histogram
fig, ax = plt.subplots()
h.stack("dataset").plot(ax=ax, stack=True, histtype="fill", linewidth=1, edgecolor="gray")
ax.set_xlim(100, 200)
ax.set_title("Leading jet $p_{T}$")
ax.legend(title="Dataset", frameon=False)

# Save to PNG
fig.savefig("leading_jet_pt_stacked_100_200.png")
plt.close(fig)
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:30
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
eb522b4f7276: Waiting
8fdd7df8e834: Waiting
2a1478e0415a: Waiting
a0831920243c: Waiting
235451d01566: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
8fdd7df8e834: Download complete
2a1478e0415a: Verifying Checksum
2a1478e0415a: Download complete
a0831920243c: Verifying Checksum
a0831920243c: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
4f4fb700ef54: Download complete
ce8f13a8c0b6: Verifying Checksum
ce8f13a8c0b6: Download complete
660b8664e6e1: Download complete
eb522b4f7276: Verifying Checksum
eb522b4f7276: Download complete
235451d01566: Verifying Checksum
235451d01566: Download complete
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


![e1255027_gpt-5_leading_jet_pt_stacked_100_200.png](img/e1255027_gpt-5_leading_jet_pt_stacked_100_200.png)
<details><summary>Usage</summary>
<div style="margin-left: 1em;">


### LLM Usage
| Step | Time (sec) | Prompt Tokens | Completion Tokens | Total Tokens | Cost (USD) |
|---|---|---|---|---|---|
| Run 1 | 110.08 | 13922 | 5525 | 19447 | $0.07 |
| **Total** | 110.08 | 13922 | 5525 | 19447 | $0.073 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 48.43 |
| **Total** | **48.43** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,110.08,13922,5525,19447,0.073,1,48.43,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 110.08 | 13922 | 5525 | 19447 | $0.073 | 1 | 48.43 | Success |
