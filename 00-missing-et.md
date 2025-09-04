# Plot the ETmiss of all events in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Fetches ETmiss and MC event weights from the requested DAOD_PHYSLITE dataset using ServiceX (NFiles=1).
- Converts ETmiss from MeV to GeV at the ServiceX level to minimize data volume.
- Uses Awkward Array for all data manipulation and Hist to build a weighted 1D histogram (50 bins, 0–200 GeV).
- Applies only the MC event weight (per guidance for a single MC dataset).
- Plots with an ATLAS-like style and saves to etmiss.png.

```python
import awkward as ak
import numpy as np
import vector
vector.register_awkward()

from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Build the ServiceX/func_adl query against PHYSLITE
base_query = FuncADLQueryPHYSLITE()

# Select per-event ETmiss (in GeV) and the MC event weight
query = (
    base_query
    .Where(lambda e: e.MissingET().Count() > 0)
    .Select(lambda e: {
        "met": e.MissingET().First().met() / 1000.0,  # MeV -> GeV
        "weight": e.EventInfo("EventInfo").mcEventWeight(0),
    })
)

# Dataset DID (Rucio)
ds_name = (
    "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE."
    "e8514_e8528_a911_s4114_r15224_r15225_p6697"
)

# Deliver the data (NFiles=1 per guidance)
all_data = to_awk(
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
        )
    )
)

data = all_data["etmiss_fetch"]

# Awkward arrays of ETmiss and weights
met = data.met        # ETmiss per event [GeV]
weights = data.weight # MC event weight

# Build a weighted histogram: 50 bins from 0 to 200 GeV
h = (
    Hist.new
    .Reg(50, 0.0, 200.0, name="met", label=r"$E_T^{\mathrm{miss}}$ [GeV]")
    .Weight()
)

# Fill using Awkward -> NumPy conversion (no Python lists)
h.fill(met=ak.to_numpy(met), weight=ak.to_numpy(weights))

# Plot
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
h.plot(histtype="fill", linewidth=1, edgecolor="gray", label="JZ2 (MC weights only)")
ax.set_xlabel(r"$E_T^{\mathrm{miss}}$ [GeV]")
ax.set_title("Missing $E_T$")
ax.legend()

# Save plot
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
2a1478e0415a: Waiting
235451d01566: Waiting
a0831920243c: Waiting
4f4fb700ef54: Waiting
ce8f13a8c0b6: Waiting
8f2cf14ad14d: Waiting
660b8664e6e1: Waiting
eb522b4f7276: Waiting
8fdd7df8e834: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
8fdd7df8e834: Verifying Checksum
8fdd7df8e834: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
2a1478e0415a: Verifying Checksum
2a1478e0415a: Download complete
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
59e22667830b: Pull complete
0f40319da806: Pull complete
081f96d6584f: Pull complete
8fdd7df8e834: Pull complete
2a1478e0415a: Pull complete
235451d01566: Pull complete
a0831920243c: Pull complete
4f4fb700ef54: Pull complete
ce8f13a8c0b6: Pull complete
8f2cf14ad14d: Verifying Checksum
8f2cf14ad14d: Download complete
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
| Run 1 | 70.78 | 14991 | 3346 | 18337 | $0.05 |
| **Total** | 70.78 | 14991 | 3346 | 18337 | $0.052 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 46.58 |
| **Total** | **46.58** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,70.78,14991,3346,18337,0.052,1,46.58,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 70.78 | 14991 | 3346 | 18337 | $0.052 | 1 | 46.58 | Success |
