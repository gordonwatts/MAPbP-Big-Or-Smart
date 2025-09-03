# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Uses ServiceX (FuncADL on PHYSLITE) to skim only events that contain at least one jet with pT ≥ 100 GeV, and only transfers jet pT’s (in GeV) and the MC event weight to minimize data movement.
- Computes the leading-jet pT per event with Awkward (sort per event, take first), then selects 100–200 GeV.
- Applies only the MC event weight (no cross-section scaling); cross-section info for these JZ samples is not provided here, so the two MC samples are stacked using MC weights only.
- Produces a stacked histogram (JZ2 + JZ3) with 50 bins over [100, 200] GeV and saves to leading_jet_pt_stacked.png.
- Runs one ServiceX delivery with two Rucio datasets and NFiles=1 for each sample (as requested in the guidelines). You can increase NFiles when you’re ready.

```python
import awkward as ak
import vector
vector.register_awkward()

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# --------------------------------------------
# Define datasets (Rucio DIDs) exactly as requested
# --------------------------------------------
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# --------------------------------------------
# Build one query for both datasets:
#  - Keep only events with >=1 jet with pt >= 100 GeV
#  - Return the jets' pt (>=100 GeV only) in GeV
#  - Return the MC event weight (mcEventWeight(0))
# --------------------------------------------
base = FuncADLQueryPHYSLITE()

query = (
    base
    .Where(lambda e: e.Jets().Where(lambda j: j.pt() / 1000.0 >= 100.0).Count() > 0)
    .Select(lambda e: {
        "jet_pt": e.Jets()
                   .Where(lambda j: j.pt() / 1000.0 >= 100.0)
                   .Select(lambda j: j.pt() / 1000.0),
        "mcw": e.EventInfo("EventInfo").First().mcEventWeight(0),
    })
)

# --------------------------------------------
# Deliver both datasets in a single call (NFiles=1 per guidelines)
# --------------------------------------------
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

# --------------------------------------------
# Compute leading-jet pT per event using Awkward
# --------------------------------------------
# Sort jet pts in each event descending and take the first (leading)
lead_pt_jz2 = ak.firsts(ak.sort(data_jz2["jet_pt"], axis=1, ascending=False))
lead_pt_jz3 = ak.firsts(ak.sort(data_jz3["jet_pt"], axis=1, ascending=False))

# MC event weights
w_jz2 = data_jz2["mcw"]
w_jz3 = data_jz3["mcw"]

# Restrict to 100–200 GeV
mask_jz2 = (lead_pt_jz2 >= 100.0) & (lead_pt_jz2 <= 200.0)
mask_jz3 = (lead_pt_jz3 >= 100.0) & (lead_pt_jz3 <= 200.0)

lead_pt_jz2_sel = lead_pt_jz2[mask_jz2]
w_jz2_sel = w_jz2[mask_jz2]

lead_pt_jz3_sel = lead_pt_jz3[mask_jz3]
w_jz3_sel = w_jz3[mask_jz3]

# --------------------------------------------
# Build a stacked histogram over [100, 200] GeV with 50 bins
# --------------------------------------------
h = (
    Hist.new
    .Reg(50, 100.0, 200.0, name="pt", label="Leading jet $p_T$ [GeV]")
    .StrCat([], name="sample", label="Sample", growth=True)
    .Weight()
)

# Fill per sample (Awkward -> NumPy for Hist fill)
h.fill(pt=ak.to_numpy(lead_pt_jz2_sel), sample="JZ2", weight=ak.to_numpy(w_jz2_sel))
h.fill(pt=ak.to_numpy(lead_pt_jz3_sel), sample="JZ3", weight=ak.to_numpy(w_jz3_sel))

# --------------------------------------------
# Plot and save
# --------------------------------------------
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
h.plot(histtype="fill", stack=True, ax=ax)
ax.set_title("Leading jet $p_T$")
ax.legend(title="Sample")
fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0/?  
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0/?  
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0/?  
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0/?  

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
2a1478e0415a: Waiting
235451d01566: Waiting
a0831920243c: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
8fdd7df8e834: Verifying Checksum
8fdd7df8e834: Download complete
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
660b8664e6e1: Verifying Checksum
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
Method First not found on object <class 'func_adl_servicex_xaodr25.xAOD.eventinfo_v1.EventInfo_v1'>
Traceback (most recent call last):
  File "/app/script.py", line 41, in <module>
    result = to_awk(
        deliver(
    ...<16 lines>...
        )
    )
  File "/work/.venv/lib/python3.13/site-packages/servicex_analysis_utils/materialization.py", line 56, in to_awk
    if not paths:
           ^^^^^
  File "/work/.venv/lib/python3.13/site-packages/servicex/servicex_client.py", line 110, in __len__
    raise data
servicex.servicex_client.ReturnValueException: Exception occurred while making ServiceX request.
Traceback (most recent call last):
  File "/work/.venv/lib/python3.13/site-packages/servicex/query_core.py", line 655, in as_files_async
    return await self.submit_and_download(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        signed_urls_only=False, expandable_progress=progress
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/work/.venv/lib/python3.13/site-packages/servicex/query_core.py", line 327, in submit_and_download
    self.request_id = await self.servicex.submit_transform(sx_request)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/.venv/lib/python3.13/site-packages/servicex/servicex_adapter.py", line 372, in submit_transform
    raise RuntimeError(
    ...<2 lines>...
    )
RuntimeError: ServiceX WebAPI Error during transformation submission: 500 - Something went wrong (Failed to generate translation code: Unable to generate a sequence from the given AST. Either there is an internal error, or you are trying to manipulate a const xAOD::EventInfo_v1 * value (expression eventinfo5609) ('cpp_variable') as a sequence (ast is: ('EventInfo')))


```


</div></details>
<details><summary>Run 2</summary>
<div style="margin-left: 1em;">


- Root cause: EventInfo is a single object, not a sequence. Calling .First() on it caused:
  - “Method First not found on object … EventInfo_v1”
  - Translator failure: “trying to manipulate a const xAOD::EventInfo_v1* value as a sequence”
- Fix: Remove .First() when accessing the MC event weight. Use e.EventInfo("EventInfo").mcEventWeight(0) directly.
- Everything else (single deliver call, NFiles=1, Awkward-only manipulation, histogram to PNG) remains as before.

```python
import awkward as ak
import vector
vector.register_awkward()

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# --------------------------------------------
# Define datasets (Rucio DIDs) exactly as requested
# --------------------------------------------
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# --------------------------------------------
# Build one query for both datasets:
#  - Keep only events with >=1 jet with pt >= 100 GeV
#  - Return the jets' pt (>=100 GeV only) in GeV
#  - Return the MC event weight (mcEventWeight(0))
# --------------------------------------------
base = FuncADLQueryPHYSLITE()

query = (
    base
    .Where(lambda e: e.Jets().Where(lambda j: j.pt() / 1000.0 >= 100.0).Count() > 0)
    .Select(lambda e: {
        "jet_pt": e.Jets()
                   .Where(lambda j: j.pt() / 1000.0 >= 100.0)
                   .Select(lambda j: j.pt() / 1000.0),
        # EventInfo is a single object; do not call .First() on it
        "mcw": e.EventInfo("EventInfo").mcEventWeight(0),
    })
)

# --------------------------------------------
# Deliver both datasets in a single call (NFiles=1 per guidelines)
# --------------------------------------------
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

# --------------------------------------------
# Compute leading-jet pT per event using Awkward
# --------------------------------------------
lead_pt_jz2 = ak.firsts(ak.sort(data_jz2["jet_pt"], axis=1, ascending=False))
lead_pt_jz3 = ak.firsts(ak.sort(data_jz3["jet_pt"], axis=1, ascending=False))

# MC event weights
w_jz2 = data_jz2["mcw"]
w_jz3 = data_jz3["mcw"]

# Restrict to 100–200 GeV
mask_jz2 = (lead_pt_jz2 >= 100.0) & (lead_pt_jz2 <= 200.0)
mask_jz3 = (lead_pt_jz3 >= 100.0) & (lead_pt_jz3 <= 200.0)

lead_pt_jz2_sel = lead_pt_jz2[mask_jz2]
w_jz2_sel = w_jz2[mask_jz2]

lead_pt_jz3_sel = lead_pt_jz3[mask_jz3]
w_jz3_sel = w_jz3[mask_jz3]

# --------------------------------------------
# Build a stacked histogram over [100, 200] GeV with 50 bins
# --------------------------------------------
h = (
    Hist.new
    .Reg(50, 100.0, 200.0, name="pt", label="Leading jet $p_T$ [GeV]")
    .StrCat([], name="sample", label="Sample", growth=True)
    .Weight()
)

# Fill per sample (Awkward -> NumPy for Hist fill)
h.fill(pt=ak.to_numpy(lead_pt_jz2_sel), sample="JZ2", weight=ak.to_numpy(w_jz2_sel))
h.fill(pt=ak.to_numpy(lead_pt_jz3_sel), sample="JZ3", weight=ak.to_numpy(w_jz3_sel))

# --------------------------------------------
# Plot and save
# --------------------------------------------
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
h.plot(histtype="fill", stack=True, ax=ax)
ax.set_title("Leading jet $p_T$")
ax.legend(title="Sample")
fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:30
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:41
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:45

```

### stderr:

```text
bash: cannot set terminal process group (-1): Inappropriate ioctl for device
bash: no job control in this shell
/work/.venv/lib/python3.13/site-packages/func_adl_servicex_xaodr25/trigger.py:127: SyntaxWarning: invalid escape sequence '\D'
  object. Close match is done as a function of $\Delta R < `dr`$.
/work/.venv/lib/python3.13/site-packages/hist/quick_construct.py:381: UserWarning: sample is a protected keyword and cannot be used as axis name
  return self.hist_class(
Traceback (most recent call last):
  File "/app/script.py", line 97, in <module>
    h.fill(pt=ak.to_numpy(lead_pt_jz2_sel), sample="JZ2", weight=ak.to_numpy(w_jz2_sel))
    ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/.venv/lib/python3.13/site-packages/hist/basehist.py", line 259, in fill
    raise TypeError(
        "All axes must be accounted for in fill, you may have used a disallowed name in the axes"
    )
TypeError: All axes must be accounted for in fill, you may have used a disallowed name in the axes

```


</div></details>
<details><summary>Run 3</summary>
<div style="margin-left: 1em;">


- The Hist error comes from using "sample" as an axis name. In Hist, "sample" is a protected keyword and cannot be used as an axis name, which triggers the warning. Because the axis name was rejected, the subsequent fill call could not match all axes ("All axes must be accounted for in fill"), causing the TypeError.
- Fix: rename the categorical axis from "sample" to a non-reserved name (e.g., "ds"), and update the fill calls to use ds="JZ2"/"JZ3". Everything else remains the same. The ServiceX query and Awkward manipulations are fine.

```python
import awkward as ak
import vector
vector.register_awkward()

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

# --------------------------------------------
# Define datasets (Rucio DIDs) exactly as requested
# --------------------------------------------
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# --------------------------------------------
# Build one query for both datasets:
#  - Keep only events with >=1 jet with pt >= 100 GeV
#  - Return the jets' pt (>=100 GeV only) in GeV
#  - Return the MC event weight (mcEventWeight(0))
# --------------------------------------------
base = FuncADLQueryPHYSLITE()

query = (
    base
    .Where(lambda e: e.Jets().Where(lambda j: j.pt() / 1000.0 >= 100.0).Count() > 0)
    .Select(lambda e: {
        "jet_pt": e.Jets()
                   .Where(lambda j: j.pt() / 1000.0 >= 100.0)
                   .Select(lambda j: j.pt() / 1000.0),
        "mcw": e.EventInfo("EventInfo").mcEventWeight(0),
    })
)

# --------------------------------------------
# Deliver both datasets in a single call (NFiles=1 per guidelines)
# --------------------------------------------
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

# --------------------------------------------
# Compute leading-jet pT per event using Awkward
# --------------------------------------------
lead_pt_jz2 = ak.firsts(ak.sort(data_jz2["jet_pt"], axis=1, ascending=False))
lead_pt_jz3 = ak.firsts(ak.sort(data_jz3["jet_pt"], axis=1, ascending=False))

# MC event weights
w_jz2 = data_jz2["mcw"]
w_jz3 = data_jz3["mcw"]

# Restrict to 100–200 GeV
mask_jz2 = (lead_pt_jz2 >= 100.0) & (lead_pt_jz2 <= 200.0)
mask_jz3 = (lead_pt_jz3 >= 100.0) & (lead_pt_jz3 <= 200.0)

lead_pt_jz2_sel = lead_pt_jz2[mask_jz2]
w_jz2_sel = w_jz2[mask_jz2]

lead_pt_jz3_sel = lead_pt_jz3[mask_jz3]
w_jz3_sel = w_jz3[mask_jz3]

# --------------------------------------------
# Build a stacked histogram over [100, 200] GeV with 50 bins
# --------------------------------------------
h = (
    Hist.new
    .Reg(50, 100.0, 200.0, name="pt", label="Leading jet $p_{T}$ [GeV]")
    .StrCat([], name="ds", label="Dataset", growth=True)  # renamed axis from 'sample' -> 'ds'
    .Weight()
)

# Fill per dataset (Awkward -> NumPy for Hist fill)
h.fill(pt=ak.to_numpy(lead_pt_jz2_sel), ds="JZ2", weight=ak.to_numpy(w_jz2_sel))
h.fill(pt=ak.to_numpy(lead_pt_jz3_sel), ds="JZ3", weight=ak.to_numpy(w_jz3_sel))

# --------------------------------------------
# Plot and save
# --------------------------------------------
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
h.plot(histtype="fill", stack=True, ax=ax)
ax.set_title("Leading jet $p_{T}$")
ax.legend(title="Dataset")
fig.savefig("leading_jet_pt_stacked.png")
plt.close(fig)
```
### stdout:

```text

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
| Run 1 | 98.71 | 14211 | 4612 | 18823 | $0.06 |
| Run 2 | 57.75 | 16462 | 2550 | 19012 | $0.05 |
| Run 3 | 55.89 | 15485 | 2366 | 17851 | $0.04 |
| **Total** | 212.35 | 46158 | 9528 | 55686 | $0.153 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 17.03 |
| Run 2 | 53.40 |
| Run 3 | 7.12 |
| **Total** | **77.55** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,212.35,46158,9528,55686,0.153,3,77.55,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 212.35 | 46158 | 9528 | 55686 | $0.153 | 3 | 77.55 | Success |
