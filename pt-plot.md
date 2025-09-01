# Plot the ETmiss of all events in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Constructed a FuncADL query on DAOD_PHYSLITE to select event-level missing transverse energy (ETmiss) and converted it to GeV in the query.
- Used ServiceX to deliver data from the specified Rucio dataset (NFiles=1, as recommended) and converted the result into an Awkward Array.
- Filled a 1D histogram (50 bins, 0–200 GeV) of ETmiss using Hist, avoiding Python lists and operating directly on Awkward/Numpy arrays.
- Plotted the histogram with mplhep’s ATLAS style and saved it to a PNG file named etmiss.png.
- Note: This example uses a single file for a quick test. When ready, you can increase NFiles yourself.

```python
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Define the Rucio dataset DID exactly as requested
ds_name = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

# Build the query: event-level ETmiss (convert MeV -> GeV)
base_query = FuncADLQueryPHYSLITE()
met_query = base_query.Select(
    lambda e: {
        "met": e.MissingET().First().met() / 1000.0,  # GeV
    }
)

# Deliver the data from ServiceX (use NFiles=1 as a quick test)
results = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="met_fetch",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,
                    Query=met_query,
                )
            ]
        )
    )
)

data = results["met_fetch"]
met = data.met  # Awkward Array of ETmiss per event (GeV)

# Create and fill histogram: 50 bins from 0 to 200 GeV
h = (
    Hist.new
        .Reg(50, 0.0, 200.0, name="met", label=r"$E_{T}^{\mathrm{miss}}$ [GeV]")
        .Int64()
)
h.fill(met=ak.to_numpy(met))

# Plot
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
h.plot(histtype="fill", linewidth=1, edgecolor="gray", label="All events")
ax.set_xlabel(r"$E_{T}^{\mathrm{miss}}$ [GeV]")
ax.set_ylabel("Events")
ax.set_title(r"Missing $E_{T}^{\mathrm{miss}}$")
ax.legend()

# Save to file
fig.savefig("etmiss.png")
plt.close(fig)
```
### stdout:

```text
met_fetch: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
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
8fdd7df8e834: Waiting
ce8f13a8c0b6: Waiting
8f2cf14ad14d: Waiting
660b8664e6e1: Waiting
2a1478e0415a: Waiting
235451d01566: Waiting
eb522b4f7276: Waiting
a0831920243c: Waiting
4f4fb700ef54: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
8fdd7df8e834: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
2a1478e0415a: Verifying Checksum
2a1478e0415a: Download complete
a0831920243c: Verifying Checksum
a0831920243c: Download complete
4f4fb700ef54: Verifying Checksum
4f4fb700ef54: Download complete
59e22667830b: Pull complete
ce8f13a8c0b6: Verifying Checksum
ce8f13a8c0b6: Download complete
0f40319da806: Pull complete
660b8664e6e1: Verifying Checksum
660b8664e6e1: Download complete
235451d01566: Verifying Checksum
235451d01566: Download complete
081f96d6584f: Pull complete
8fdd7df8e834: Pull complete
2a1478e0415a: Pull complete
eb522b4f7276: Download complete
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
| Run 1 | 40.42 | 13846 | 3114 | 16960 | $0.05 |
| **Total** | 40.42 | 13846 | 3114 | 16960 | $0.048 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 46.01 |
| **Total** | **46.01** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,40.42,13846,3114,16960,0.048,1,46.01,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 40.42 | 13846 | 3114 | 16960 | $0.048 | 1 | 46.01 | Success |
