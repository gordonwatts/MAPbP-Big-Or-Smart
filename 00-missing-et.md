# Plot the ETmiss of all events in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Fetched only the quantities needed (missing transverse energy and MC event weight) from the requested Rucio dataset using ServiceX, converting MeV to GeV inside the query.
- Used Awkward Array to hold and manipulate the returned data (no Python lists).
- Applied per-event MC weights (EventInfo.mcEventWeight(0)) when filling the histogram. No cross section or luminosity scaling was applied since only a single MC dataset was requested.
- Plotted the ETmiss distribution with Hist/mplhep and saved it to a PNG file.
- Processed only 1 file (NFiles=1) to keep turnaround fast; you can increase this when ready.

```python
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Build the query against PHYSLITE:
# - Convert MeV -> GeV in the query (divide by 1000.0)
base_query = FuncADLQueryPHYSLITE()
query = (
    base_query
    .Select(lambda e: {
        "met": e.MissingET().First().met() / 1000.0,
        "mcw": e.EventInfo("EventInfo").mcEventWeight(0),
    })
)

# Dataset DID provided by the user (copy carefully!)
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv."
    "DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
)

# Deliver the data (use NFiles=1 for a quick turn-around)
results = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="etmiss_fetch",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,
                    Query=query,
                )
            ]
        ),
    )
)

data = results["etmiss_fetch"]

# Awkward arrays holding the event-level ETmiss and per-event MC weight
met = data.met     # [GeV]
mcw = data.mcw     # per-event MC weight

# Build a weighted histogram for ETmiss
# Binning & range chosen as a reasonable default for MET
h = (
    Hist.new
        .Reg(50, 0.0, 200.0, name="met", label=r"$E_T^{miss}$ [GeV]")
        .Weight()
)

# Fill with weights; convert to NumPy for Hist backend while staying array-based (no Python lists)
h.fill(met=ak.to_numpy(met), weight=ak.to_numpy(mcw))

# Plot using ATLAS-like style
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()

h.plot(histtype="fill", linewidth=1, edgecolor="gray", label="MC (JZ2), weighted")
ax.legend()
ax.set_xlabel(r"$E_T^{miss}$ [GeV]")
ax.set_ylabel("Events")
ax.set_title(r"Missing $E_T$")

# Optional: annotate what weights were applied
ax.text(
    0.98, 0.95,
    "Weights: mcEventWeight(0)\nNo XS/L scaling",
    transform=ax.transAxes,
    ha="right", va="top", fontsize=10
)

fig.savefig("etmiss.png")
plt.close(fig)
```
### stdout:

```text
etmiss_fetch: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:21
               Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
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
235451d01566: Waiting
a0831920243c: Waiting
4f4fb700ef54: Waiting
ce8f13a8c0b6: Waiting
660b8664e6e1: Waiting
8f2cf14ad14d: Waiting
8fdd7df8e834: Waiting
2a1478e0415a: Waiting
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
a0831920243c: Verifying Checksum
a0831920243c: Download complete
4f4fb700ef54: Download complete
ce8f13a8c0b6: Verifying Checksum
ce8f13a8c0b6: Download complete
660b8664e6e1: Verifying Checksum
660b8664e6e1: Download complete
eb522b4f7276: Download complete
235451d01566: Verifying Checksum
235451d01566: Download complete
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


![2fc7e91d_gpt-5_etmiss.png](img/2fc7e91d_gpt-5_etmiss.png)
<details><summary>Usage</summary>
<div style="margin-left: 1em;">


### LLM Usage
| Step | Time (sec) | Prompt Tokens | Completion Tokens | Total Tokens | Cost (USD) |
|---|---|---|---|---|---|
| Run 1 | 55.44 | 14646 | 2838 | 17484 | $0.05 |
| **Total** | 55.44 | 14646 | 2838 | 17484 | $0.047 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 43.41 |
| **Total** | **43.41** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,55.44,14646,2838,17484,0.047,1,43.41,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 55.44 | 14646 | 2838 | 17484 | $0.047 | 1 | 43.41 | Success |
