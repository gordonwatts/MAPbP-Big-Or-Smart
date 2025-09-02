# Compare the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 between 100 and 200 GeV.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Use ServiceX (FuncADL, PHYSLITE) to fetch, in one call, only the per-event jet pT lists from both rucio datasets (NFiles=1 as required).
- Convert MeV→GeV at query time; keep event structure so we can compute the leading jet pT with Awkward (ak.argmax along axis=1).
- Select only events with a valid leading jet, then restrict the leading-jet pT to 100–200 GeV for comparison.
- Fill two Hist histograms (50 bins, 100–200 GeV), and plot them overlaid with ATLAS style.
- Save the comparison plot to a PNG file. If you want more than 1 file, run locally with more files by changing NFiles yourself.

```python
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

def main():
    # Use ATLAS-like plotting style
    plt.style.use(hep.style.ATLAS)

    # Base query against PHYSLITE: per-event list of jet pT in GeV
    base_query = FuncADLQueryPHYSLITE()
    query = (
        base_query
        .Select(lambda e: {
            "jet_pt": e.Jets().Select(lambda j: j.pt() / 1000.0),  # GeV
        })
    )

    # Rucio DIDs (copied exactly as requested)
    ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
    ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

    # Deliver data for both datasets in a single call (NFiles=1 as per guidance)
    results = to_awk(
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
            ),
        )
    )

    # Extract jet pT lists (Awkward) per event
    jets_pt_jz2 = results["JZ2"].jet_pt
    jets_pt_jz3 = results["JZ3"].jet_pt

    # Compute leading jet pT per event using argmax pattern (handles empty events -> None)
    idx_jz2 = ak.argmax(jets_pt_jz2, axis=1, keepdims=True)
    lead_pt_jz2 = ak.firsts(jets_pt_jz2[idx_jz2])

    idx_jz3 = ak.argmax(jets_pt_jz3, axis=1, keepdims=True)
    lead_pt_jz3 = ak.firsts(jets_pt_jz3[idx_jz3])

    # Remove events with no jets (None) and keep 100–200 GeV window
    valid_jz2 = ~ak.is_none(lead_pt_jz2, axis=0)
    valid_jz3 = ~ak.is_none(lead_pt_jz3, axis=0)

    lead_pt_jz2 = lead_pt_jz2[valid_jz2]
    lead_pt_jz3 = lead_pt_jz3[valid_jz3]

    in_window_jz2 = (lead_pt_jz2 >= 100.0) & (lead_pt_jz2 <= 200.0)
    in_window_jz3 = (lead_pt_jz3 >= 100.0) & (lead_pt_jz3 <= 200.0)

    lead_pt_jz2_win = lead_pt_jz2[in_window_jz2]
    lead_pt_jz3_win = lead_pt_jz3[in_window_jz3]

    # Build histograms (50 bins from 100 to 200 GeV)
    from hist import Hist

    h_jz2 = Hist.new.Reg(50, 100.0, 200.0, name="pt", label="Leading jet $p_{T}$ [GeV]").Int64()
    h_jz3 = Hist.new.Reg(50, 100.0, 200.0, name="pt", label="Leading jet $p_{T}$ [GeV]").Int64()

    # Fill histograms (convert to NumPy for filling; still no Python lists)
    h_jz2.fill(pt=ak.to_numpy(lead_pt_jz2_win))
    h_jz3.fill(pt=ak.to_numpy(lead_pt_jz3_win))

    # Plot overlaid comparison
    fig, ax = plt.subplots()
    h_jz2.plot(histtype="fill", alpha=0.5, label="JZ2")
    h_jz3.plot(histtype="fill", alpha=0.5, label="JZ3")
    ax.set_xlim(100.0, 200.0)
    ax.set_xlabel("Leading jet $p_{T}$ [GeV]")
    ax.set_ylabel("Events")
    ax.legend()
    ax.set_title("Leading jet $p_{T}$ (100–200 GeV)")

    fig.savefig("leading_jet_pt_JZ2_vs_JZ3.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:25
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:20
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
4f4fb700ef54: Waiting
ce8f13a8c0b6: Waiting
eb522b4f7276: Waiting
8f2cf14ad14d: Waiting
660b8664e6e1: Waiting
235451d01566: Waiting
2a1478e0415a: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
8fdd7df8e834: Verifying Checksum
8fdd7df8e834: Download complete
2a1478e0415a: Verifying Checksum
2a1478e0415a: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
a0831920243c: Verifying Checksum
a0831920243c: Download complete
ce8f13a8c0b6: Verifying Checksum
ce8f13a8c0b6: Download complete
4f4fb700ef54: Verifying Checksum
4f4fb700ef54: Download complete
660b8664e6e1: Verifying Checksum
660b8664e6e1: Download complete
eb522b4f7276: Verifying Checksum
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


![44e91a77_gpt-5_leading_jet_pt_JZ2_vs_JZ3.png](img/44e91a77_gpt-5_leading_jet_pt_JZ2_vs_JZ3.png)
<details><summary>Usage</summary>
<div style="margin-left: 1em;">


### LLM Usage
| Step | Time (sec) | Prompt Tokens | Completion Tokens | Total Tokens | Cost (USD) |
|---|---|---|---|---|---|
| Run 1 | 108.61 | 13919 | 4691 | 18610 | $0.06 |
| **Total** | 108.61 | 13919 | 4691 | 18610 | $0.064 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 44.44 |
| **Total** | **44.44** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,108.61,13919,4691,18610,0.064,1,44.44,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 108.61 | 13919 | 4691 | 18610 | $0.064 | 1 | 44.44 | Success |
