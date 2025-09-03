# Make stacked histogram of the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Uses ServiceX (single deliver call) to read both Rucio datasets and only return per-event jet pT values (in GeV) with a ServiceX-side jet pT > 50 GeV filter to reduce data volume.
- Computes the leading (highest pT) jet per event using Awkward (argmax trick with keepdims=True), then selects events where 100 GeV ≤ pT ≤ 200 GeV.
- Builds two Hist histograms (one per dataset) and makes a stacked plot using mplhep; saves to PNG.
- Only 1 file per dataset is processed (NFiles=1) for speed; expand locally later if desired.
- No cross-section scaling is applied; MC event weights are not applied here (not fetched). If you want nominal MC weights, we can extend the query to include EventInfo weights and apply them.

```python
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist
from servicex import deliver, ServiceXSpec, Sample, dataset
from servicex_analysis_utils import to_awk
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE

# Use ATLAS-like plotting style
plt.style.use(hep.style.ATLAS)

# Rucio dataset names (copied exactly as provided)
ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Build a ServiceX query:
# - For each event, return a list of jet pt values (in GeV).
# - Apply a ServiceX-side filter to keep only jets with pt > 50 GeV to reduce data volume.
base = FuncADLQueryPHYSLITE()
query = (
    base
    .Select(lambda e: e.Jets().Where(lambda j: j.pt()/1000.0 > 50.0))
    .Select(lambda jets: {
        "jet_pt": jets.Select(lambda j: j.pt()/1000.0),  # GeV
    })
)

# Deliver from ServiceX for both datasets at once (NFiles=1 for speed)
sx_out = to_awk(
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

# Access the awkward arrays per sample
arr_jz2 = sx_out["JZ2"]
arr_jz3 = sx_out["JZ3"]

# Helper: compute leading jet pt per event from a jagged list of jet pts
def leading_pt_per_event(jet_pt_nested):
    # argmax per event (axis=1); keepdims=True so we can slice safely
    idx = ak.argmax(jet_pt_nested, axis=1, keepdims=True)
    # Slice to get leading pt; will yield None for events with no jets
    leading = jet_pt_nested[idx]
    # Remove the extra nesting introduced by keepdims=True
    leading = ak.flatten(leading, axis=1)
    # Drop None (events with no surviving jets)
    leading = leading[~ak.is_none(leading)]
    return leading

lead_jz2 = leading_pt_per_event(arr_jz2["jet_pt"])
lead_jz3 = leading_pt_per_event(arr_jz3["jet_pt"])

# Keep only leading jet pT in [100, 200] GeV
def in_range(x, lo=100.0, hi=200.0):
    return x[(x >= lo) & (x <= hi)]

lead_jz2_sel = in_range(lead_jz2, 100.0, 200.0)
lead_jz3_sel = in_range(lead_jz3, 100.0, 200.0)

# Build histograms (50 bins between 100 and 200 GeV)
h_jz2 = (
    Hist.new
    .Reg(50, 100.0, 200.0, name="jet_pt", label="Leading jet $p_{T}$ [GeV]")
    .Weight()
)
h_jz3 = (
    Hist.new
    .Reg(50, 100.0, 200.0, name="jet_pt", label="Leading jet $p_{T}$ [GeV]")
    .Weight()
)

# No event-weighting applied here (weight=1.0). If you want MC nominal weights,
# extend the query to fetch EventInfo weights and pass them here.
h_jz2.fill(jet_pt=lead_jz2_sel, weight=1.0)
h_jz3.fill(jet_pt=lead_jz3_sel, weight=1.0)

# Make a stacked plot
fig, ax = plt.subplots()

# mplhep can stack multiple Hist objects
hep.histplot(
    [h_jz2, h_jz3],
    ax=ax,
    histtype="fill",
    stack=True,
    label=["JZ2", "JZ3"],
    edgecolor="black",
    linewidth=0.8,
)

ax.set_xlabel("Leading jet $p_{T}$ [GeV]")
ax.set_ylabel("Events")
ax.set_title("Leading jet $p_{T}$ (100–200 GeV)")
ax.legend()

fig.tight_layout()
fig.savefig("leading_jet_pt_100_200_stacked.png")
plt.close(fig)

print("Wrote: leading_jet_pt_100_200_stacked.png")
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:30
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
Wrote: leading_jet_pt_100_200_stacked.png
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
8fdd7df8e834: Waiting
2a1478e0415a: Waiting
eb522b4f7276: Waiting
a0831920243c: Waiting
235451d01566: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
8fdd7df8e834: Verifying Checksum
8fdd7df8e834: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
a0831920243c: Verifying Checksum
a0831920243c: Download complete
4f4fb700ef54: Verifying Checksum
4f4fb700ef54: Download complete
2a1478e0415a: Verifying Checksum
2a1478e0415a: Download complete
ce8f13a8c0b6: Verifying Checksum
ce8f13a8c0b6: Download complete
235451d01566: Verifying Checksum
235451d01566: Download complete
eb522b4f7276: Verifying Checksum
eb522b4f7276: Download complete
660b8664e6e1: Verifying Checksum
660b8664e6e1: Download complete
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


![e1255027_gpt-5_leading_jet_pt_100_200_stacked.png](img/e1255027_gpt-5_leading_jet_pt_100_200_stacked.png)
<details><summary>Usage</summary>
<div style="margin-left: 1em;">


### LLM Usage
| Step | Time (sec) | Prompt Tokens | Completion Tokens | Total Tokens | Cost (USD) |
|---|---|---|---|---|---|
| Run 1 | 143.26 | 14088 | 5429 | 19517 | $0.07 |
| **Total** | 143.26 | 14088 | 5429 | 19517 | $0.072 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 48.25 |
| **Total** | **48.25** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,143.26,14088,5429,19517,0.072,1,48.25,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 143.26 | 14088 | 5429 | 19517 | $0.072 | 1 | 48.25 | Success |
