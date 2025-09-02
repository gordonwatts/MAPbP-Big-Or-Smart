# Compare the pT of the highest pT jet in each event in mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697 and mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Use ServiceX (FuncADL, PHYSLITE) to fetch, in one delivery, per-event jet pT lists from the two requested DAOD_PHYSLITE rucio datasets; convert MeV to GeV at the source.
- Use Awkward to find the leading-jet pT per event via ak.argmax with keepdims and slicing; remove events with zero jets.
- Fill two Hist 1D histograms (50 bins, 0–300 GeV) for the leading-jet pT and overlay them.
- Style with mplhep ATLAS and save the comparison as a PNG.
- Only NFiles=1 is used (recommended for testing). When you are ready, increase NFiles yourself to process more files.

```python
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

def main():
    # Build the query: per-event list of jet pT in GeV
    base_query = FuncADLQueryPHYSLITE()
    jet_pts_query = (
        base_query
        .Select(lambda e: e.Jets())
        .Select(lambda jets: {
            "jet_pt": jets.Select(lambda j: j.pt() / 1000.0),  # GeV
        })
    )

    # Rucio dataset names (exactly as requested)
    ds_jz2 = "mc23_13p6TeV:mc23_13p6TeV.801167.Py8EG_A14NNPDF23LO_jj_JZ2.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"
    ds_jz3 = "mc23_13p6TeV:mc23_13p6TeV.801168.Py8EG_A14NNPDF23LO_jj_JZ3.deriv.DAOD_PHYSLITE.e8514_e8528_a911_s4114_r15224_r15225_p6697"

    # Single ServiceX delivery for both datasets (NFiles=1 per instructions)
    result = to_awk(
        deliver(
            ServiceXSpec(
                Sample=[
                    Sample(
                        Name="JZ2",
                        Dataset=dataset.Rucio(ds_jz2),
                        NFiles=1,
                        Query=jet_pts_query,
                    ),
                    Sample(
                        Name="JZ3",
                        Dataset=dataset.Rucio(ds_jz3),
                        NFiles=1,
                        Query=jet_pts_query,
                    ),
                ]
            )
        )
    )

    # Extract Awkward arrays of per-event jet pT lists
    jets_jz2 = result["JZ2"].jet_pt
    jets_jz3 = result["JZ3"].jet_pt

    # Leading-jet pT per event using argmax trick (handles empty events -> None)
    jz2_idx = ak.argmax(jets_jz2, axis=1, keepdims=True)
    jz3_idx = ak.argmax(jets_jz3, axis=1, keepdims=True)

    lead_pt_jz2 = ak.flatten(jets_jz2[jz2_idx])
    lead_pt_jz3 = ak.flatten(jets_jz3[jz3_idx])

    # Drop events with no jets (None)
    lead_pt_jz2 = lead_pt_jz2[~ak.is_none(lead_pt_jz2)]
    lead_pt_jz3 = lead_pt_jz3[~ak.is_none(lead_pt_jz3)]

    # Create histograms
    h_jz2 = Hist.new.Reg(50, 0.0, 300.0, name="pt", label="Leading jet $p_{T}$ [GeV]").Int64()
    h_jz3 = Hist.new.Reg(50, 0.0, 300.0, name="pt", label="Leading jet $p_{T}$ [GeV]").Int64()

    # Fill (convert to NumPy for safety)
    h_jz2.fill(pt=ak.to_numpy(lead_pt_jz2))
    h_jz3.fill(pt=ak.to_numpy(lead_pt_jz3))

    # Plot
    plt.style.use(hep.style.ATLAS)
    fig, ax = plt.subplots()
    h_jz2.plot(ax=ax, histtype="step", linewidth=1.5, label="JZ2")
    h_jz3.plot(ax=ax, histtype="step", linewidth=1.5, label="JZ3")
    ax.set_xlabel("Leading jet $p_{T}$ [GeV]")
    ax.set_ylabel("Events")
    ax.set_title("Leading jet $p_{T}$")
    ax.legend()

    fig.savefig("leading_jet_pt_comparison.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
```
### stdout:

```text
JZ2: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:31
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:35
JZ3: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:31
      Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:35
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
8fdd7df8e834: Waiting
ce8f13a8c0b6: Waiting
eb522b4f7276: Waiting
8f2cf14ad14d: Waiting
660b8664e6e1: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
8fdd7df8e834: Verifying Checksum
8fdd7df8e834: Download complete
2a1478e0415a: Download complete
a0831920243c: Verifying Checksum
a0831920243c: Download complete
59e22667830b: Download complete
4f4fb700ef54: Verifying Checksum
4f4fb700ef54: Download complete
ce8f13a8c0b6: Verifying Checksum
ce8f13a8c0b6: Download complete
660b8664e6e1: Verifying Checksum
660b8664e6e1: Download complete
235451d01566: Verifying Checksum
235451d01566: Download complete
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
/work/.venv/lib/python3.13/site-packages/mplhep/utils.py:741: UserWarning: Integer weights indicate poissonian data. Will calculate Garwood interval if ``scipy`` is installed. Otherwise errors will be set to ``sqrt(w2)``.
  self.errors()
/work/.venv/lib/python3.13/site-packages/mplhep/utils.py:741: UserWarning: Integer weights indicate poissonian data. Will calculate Garwood interval if ``scipy`` is installed. Otherwise errors will be set to ``sqrt(w2)``.
  self.errors()

```


</div></details>


![f13afc1a_gpt-5_leading_jet_pt_comparison.png](img/f13afc1a_gpt-5_leading_jet_pt_comparison.png)
<details><summary>Usage</summary>
<div style="margin-left: 1em;">


### LLM Usage
| Step | Time (sec) | Prompt Tokens | Completion Tokens | Total Tokens | Cost (USD) |
|---|---|---|---|---|---|
| Run 1 | 83.68 | 13911 | 3463 | 17374 | $0.05 |
| **Total** | 83.68 | 13911 | 3463 | 17374 | $0.052 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 66.65 |
| **Total** | **66.65** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,83.68,13911,3463,17374,0.052,1,66.65,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 83.68 | 13911 | 3463 | 17374 | $0.052 | 1 | 66.65 | Success |
