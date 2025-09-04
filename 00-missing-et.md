# Plot the ETmiss of all events in the rucio dataset mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Fetches Missing ET (ETmiss) and MC event weight from the requested rucio dataset using ServiceX with a single delivery call (NFiles=1 by design; increase when you’re ready).
- Converts MeV to GeV at the ServiceX level before transferring data.
- Uses Awkward Array end-to-end (no Python lists) and fills a weighted histogram with Hist.
- Applies MC event weights only (no luminosity scaling or cross-section normalization).
- Produces and saves a 1D histogram of ETmiss to met.png with ATLAS style, 50 bins over 0–200 GeV.

```python
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

def main():
    # Use ATLAS plotting style
    plt.style.use(hep.style.ATLAS)

    # Dataset DID (as requested)
    ds_name = ("mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv."
               "DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697")

    # Build the ServiceX query against PHYSLITE
    # - Convert MeV -> GeV for MET at the source
    # - Fetch per-event MC weight
    base_query = FuncADLQueryPHYSLITE()
    query = base_query.Select(
        lambda e: {
            "met": e.MissingET().First().met() / 1000.0,  # GeV
            "weight": e.EventInfo("EventInfo").mcEventWeight(0),
        }
    )

    # Deliver from ServiceX (NFiles=1 to test quickly; increase later if needed)
    data_dict = to_awk(
        deliver(
            ServiceXSpec(
                Sample=[
                    Sample(
                        Name="met_fetch",
                        Dataset=dataset.Rucio(ds_name),
                        NFiles=1,
                        Query=query,
                    )
                ]
            )
        )
    )

    data = data_dict["met_fetch"]
    met = data.met          # Awkward 1D array of ETmiss [GeV]
    weights = data.weight   # Awkward 1D array of per-event MC weights

    # Define a weighted histogram for ETmiss
    h_met = (
        Hist.new
        .Reg(50, 0.0, 200.0, name="met", label=r"$E_T^{\mathrm{miss}}$ [GeV]")
        .Weight()  # allow weighted entries (stores sumw and sumw2)
    )

    # Fill histogram with awkward arrays (no Python lists)
    h_met.fill(met=met, weight=weights)

    # Plot and save
    fig, ax = plt.subplots()
    h_met.plot(ax=ax, histtype="fill", linewidth=1, edgecolor="gray", label="MC (weighted)")
    ax.legend()
    ax.set_xlabel(r"$E_T^{\mathrm{miss}}$ [GeV]")
    ax.set_title("Missing $E_T$")
    fig.savefig("met.png")
    plt.close(fig)

    # Optional: simple summary on stdout
    print(f"Filled histogram with {len(met)} events (weighted). Output: met.png")
    print("Notes:")
    print("- Units converted to GeV at the source (ServiceX).")
    print("- Applied per-event MC weights only (no lumi or cross-section scaling).")
    print("- Processed NFiles=1; increase for full statistics when ready.")

if __name__ == "__main__":
    main()
```
### stdout:

```text
met_fetch: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
            Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
Filled histogram with 30000 events (weighted). Output: met.png
Notes:
- Units converted to GeV at the source (ServiceX).
- Applied per-event MC weights only (no lumi or cross-section scaling).
- Processed NFiles=1; increase for full statistics when ready.
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
ce8f13a8c0b6: Waiting
8f2cf14ad14d: Waiting
660b8664e6e1: Waiting
eb522b4f7276: Waiting
8fdd7df8e834: Waiting
a0831920243c: Waiting
2a1478e0415a: Waiting
235451d01566: Waiting
4f4fb700ef54: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
8fdd7df8e834: Verifying Checksum
8fdd7df8e834: Download complete
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


![2fc7e91d_gpt-5_met.png](img/2fc7e91d_gpt-5_met.png)
<details><summary>Usage</summary>
<div style="margin-left: 1em;">


### LLM Usage
| Step | Time (sec) | Prompt Tokens | Completion Tokens | Total Tokens | Cost (USD) |
|---|---|---|---|---|---|
| Run 1 | 76.02 | 14992 | 2694 | 17686 | $0.05 |
| **Total** | 76.02 | 14992 | 2694 | 17686 | $0.046 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 44.24 |
| **Total** | **44.24** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,76.02,14992,2694,17686,0.046,1,44.24,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 76.02 | 14992 | 2694 | 17686 | $0.046 | 1 | 44.24 | Success |
