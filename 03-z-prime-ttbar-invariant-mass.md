# Plot the reconstructed top quark pair mass in events with one charged lepton in the rucio dataset mc20_13TeV:mc20_13TeV.301333.Pythia8EvtGen_A14NNPDF23LO_zprime3000_tt.deriv.DAOD_PHYSLITE.e3723_s3681_r13167_p6026.

## Model gpt-5

<details><summary>Run 1</summary>
<div style="margin-left: 1em;">


- Used ServiceX (FuncADL on PHYSLITE) to skim only needed variables from the requested Rucio dataset: mc20_13TeV:mc20_13TeV.301333.Pythia8EvtGen_A14NNPDF23LO_zprime3000_tt.deriv.DAOD_PHYSLITE.e3723_s3681_r13167_p6026
- Selected semileptonic events: exactly one lepton (electron or muon) with pT > 25 GeV and |η| < 2.5, and at least two jets with pT > 25 GeV and |η| < 2.5; retrieved jets, MET (mpx, mpy), and the MC event weight
- Converted all momenta and energies from MeV to GeV at query time
- Reconstructed the neutrino longitudinal momentum pz by imposing the W-mass constraint (mW = 80.379 GeV); chose the solution with smaller |pz|
- Built four-vectors with Awkward+Vector and formed a proxy for the tt̄ system: p(tt̄) = plep + pν + sum(pjets); then m(tt̄) = |p(tt̄)|
- Filled a weighted histogram (weights = mcEventWeight only, since a single MC dataset is used) and saved the plot to mtt_semilep.png
- Only one file was processed from the dataset (NFiles=1) to demonstrate the workflow; increase NFiles yourself when ready

```python
import awkward as ak
import numpy as np
import vector
vector.register_awkward()

import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Dataset DID (exactly as requested)
ds_name = "mc20_13TeV:mc20_13TeV.301333.Pythia8EvtGen_A14NNPDF23LO_zprime3000_tt.deriv.DAOD_PHYSLITE.e3723_s3681_r13167_p6026"

# Build the query
base = FuncADLQueryPHYSLITE()

# Event selection: exactly one lepton (e or mu) with pT>25 GeV and |eta|<2.5
# and at least two jets with pT>25 GeV and |eta|<2.5
query = (
    base
    .Where(
        lambda e:
        (
            e.Electrons()
             .Where(lambda el: (el.pt()/1000.0 > 25.0) and (abs(el.eta()) < 2.5))
             .Count()
            +
            e.Muons()
             .Where(lambda mu: (mu.pt()/1000.0 > 25.0) and (abs(mu.eta()) < 2.5))
             .Count()
        ) == 1
    )
    .Where(
        lambda e:
        e.Jets()
         .Where(lambda j: (j.pt()/1000.0 > 25.0) and (abs(j.eta()) < 2.5))
         .Count() >= 2
    )
    .Select(
        lambda e: {
            "ele": e.Electrons()
                    .Where(lambda el: (el.pt()/1000.0 > 25.0) and (abs(el.eta()) < 2.5)),
            "mu":  e.Muons()
                    .Where(lambda mu: (mu.pt()/1000.0 > 25.0) and (abs(mu.eta()) < 2.5)),
            "jets": e.Jets()
                     .Where(lambda j: (j.pt()/1000.0 > 25.0) and (abs(j.eta()) < 2.5)),
            "met":  e.MissingET(),
            "mc_w": e.EventInfo("EventInfo").mcEventWeight(0),
        }
    )
    .Select(
        lambda cols: {
            "ele_pt":  cols.ele.Select(lambda x: x.pt()/1000.0),
            "ele_eta": cols.ele.Select(lambda x: x.eta()),
            "ele_phi": cols.ele.Select(lambda x: x.phi()),

            "mu_pt":  cols.mu.Select(lambda x: x.pt()/1000.0),
            "mu_eta": cols.mu.Select(lambda x: x.eta()),
            "mu_phi": cols.mu.Select(lambda x: x.phi()),

            "jet_pt":  cols.jets.Select(lambda j: j.pt()/1000.0),
            "jet_eta": cols.jets.Select(lambda j: j.eta()),
            "jet_phi": cols.jets.Select(lambda j: j.phi()),
            "jet_m":   cols.jets.Select(lambda j: j.m()/1000.0),

            "met_px": cols.met.First().mpx()/1000.0,
            "met_py": cols.met.First().mpy()/1000.0,

            "mc_w": cols.mc_w,
        }
    )
)

# Execute the query (only 1 file to keep this light; increase yourself if needed)
result = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="ttbar_mass",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,
                    Query=query,
                )
            ]
        )
    )
)

data = result["ttbar_mass"]

# Build a single lepton per event (exactly one by construction)
has_e = ak.num(data.ele_pt, axis=1) == 1

lep_pt  = ak.where(has_e, ak.firsts(data.ele_pt), ak.firsts(data.mu_pt))
lep_eta = ak.where(has_e, ak.firsts(data.ele_eta), ak.firsts(data.mu_eta))
lep_phi = ak.where(has_e, ak.firsts(data.ele_phi), ak.firsts(data.mu_phi))
# Assign lepton mass based on flavor
lep_mass = ak.where(has_e, ak.ones_like(lep_pt)*0.000511, ak.ones_like(lep_pt)*0.105658)

lep = ak.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "mass": lep_mass}, with_name="Momentum4D")

# MET components (GeV)
met_px = data.met_px
met_py = data.met_py
met2   = met_px**2 + met_py**2

# Solve neutrino pz with W-mass constraint
mW = 80.379  # GeV
a = (mW**2)/2.0 + lep.px*met_px + lep.py*met_py
A = lep.E**2 - lep.pz**2
disc = a**2 - A*met2
disc_clipped = ak.where(disc < 0.0, 0.0, disc)

pz1 = (a*lep.pz + lep.E*ak.sqrt(disc_clipped)) / A
pz2 = (a*lep.pz - lep.E*ak.sqrt(disc_clipped)) / A
use_first = ak.abs(pz1) < ak.abs(pz2)
nu_pz = ak.where(use_first, pz1, pz2)
nu_E  = ak.sqrt(met2 + nu_pz**2)

nu = ak.zip({"px": met_px, "py": met_py, "pz": nu_pz, "E": nu_E}, with_name="Momentum4D")

# Jets four-vectors
jets = ak.zip(
    {"pt": data.jet_pt, "eta": data.jet_eta, "phi": data.jet_phi, "mass": data.jet_m},
    with_name="Momentum4D",
)

# Sum jets per event (at least two jets per event by selection)
jets_sum = ak.sum(jets, axis=1)

# Reconstructed ttbar 4-vector and mass
ttbar = lep + nu + jets_sum
m_tt = ttbar.mass  # GeV

# Histogram (weighted by MC event weight only)
weights = data.mc_w

h = (
    Hist.new.Reg(50, 0.0, 4000.0, name="mtt", label="Reconstructed $m_{t\\bar{t}}$ [GeV]")
        .Weight()
)
h.fill(mtt=ak.to_numpy(m_tt), weight=ak.to_numpy(weights))

# Plot and save
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
h.plot(histtype="fill", linewidth=1, edgecolor="gray", label="MC (weighted)")
ax.legend()
ax.set_xlabel("Reconstructed $m_{t\\bar{t}}$ [GeV]")
ax.set_title("Semileptonic $t\\bar{t}$")
fig.savefig("mtt_semilep.png")
plt.close(fig)
```
### stdout:

```text
ttbar_mass: Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:31
             Download ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 00:35

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
660b8664e6e1: Waiting
eb522b4f7276: Waiting
8f2cf14ad14d: Waiting
235451d01566: Waiting
2a1478e0415a: Waiting
a0831920243c: Waiting
8fdd7df8e834: Waiting
0f40319da806: Verifying Checksum
0f40319da806: Download complete
081f96d6584f: Verifying Checksum
081f96d6584f: Download complete
8fdd7df8e834: Verifying Checksum
8fdd7df8e834: Download complete
2a1478e0415a: Verifying Checksum
2a1478e0415a: Download complete
a0831920243c: Verifying Checksum
a0831920243c: Download complete
59e22667830b: Verifying Checksum
59e22667830b: Download complete
4f4fb700ef54: Verifying Checksum
4f4fb700ef54: Download complete
ce8f13a8c0b6: Verifying Checksum
ce8f13a8c0b6: Download complete
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
Traceback (most recent call last):
  File "/app/script.py", line 118, in <module>
    pz1 = (a*lep.pz + lep.E*ak.sqrt(disc_clipped)) / A
                            ^^^^^^^
AttributeError: module 'awkward' has no attribute 'sqrt'. Did you mean: 'sort'?

```


</div></details>
<details><summary>Run 2</summary>
<div style="margin-left: 1em;">


- The immediate failure is due to calling ak.sqrt, which does not exist in Awkward. NumPy’s ufuncs (e.g., np.sqrt) are dispatched to Awkward arrays and should be used instead.
- A second latent issue is the use of ak.abs. Awkward does not provide ak.abs; the built-in Python abs works on Awkward arrays and should be used.
- The rest of the analysis logic and ServiceX usage are fine; only these two function calls need correction to allow the script to run.

```python
import awkward as ak
import numpy as np
import vector
vector.register_awkward()

import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk
from servicex import deliver, ServiceXSpec, Sample, dataset

# Dataset DID (exactly as requested)
ds_name = "mc20_13TeV:mc20_13TeV.301333.Pythia8EvtGen_A14NNPDF23LO_zprime3000_tt.deriv.DAOD_PHYSLITE.e3723_s3681_r13167_p6026"

# Build the query
base = FuncADLQueryPHYSLITE()

# Event selection: exactly one lepton (e or mu) with pT>25 GeV and |eta|<2.5
# and at least two jets with pT>25 GeV and |eta|<2.5
query = (
    base
    .Where(
        lambda e:
        (
            e.Electrons()
             .Where(lambda el: (el.pt()/1000.0 > 25.0) and (abs(el.eta()) < 2.5))
             .Count()
            +
            e.Muons()
             .Where(lambda mu: (mu.pt()/1000.0 > 25.0) and (abs(mu.eta()) < 2.5))
             .Count()
        ) == 1
    )
    .Where(
        lambda e:
        e.Jets()
         .Where(lambda j: (j.pt()/1000.0 > 25.0) and (abs(j.eta()) < 2.5))
         .Count() >= 2
    )
    .Select(
        lambda e: {
            "ele": e.Electrons()
                    .Where(lambda el: (el.pt()/1000.0 > 25.0) and (abs(el.eta()) < 2.5)),
            "mu":  e.Muons()
                    .Where(lambda mu: (mu.pt()/1000.0 > 25.0) and (abs(mu.eta()) < 2.5)),
            "jets": e.Jets()
                     .Where(lambda j: (j.pt()/1000.0 > 25.0) and (abs(j.eta()) < 2.5)),
            "met":  e.MissingET(),
            "mc_w": e.EventInfo("EventInfo").mcEventWeight(0),
        }
    )
    .Select(
        lambda cols: {
            "ele_pt":  cols.ele.Select(lambda x: x.pt()/1000.0),
            "ele_eta": cols.ele.Select(lambda x: x.eta()),
            "ele_phi": cols.ele.Select(lambda x: x.phi()),

            "mu_pt":  cols.mu.Select(lambda x: x.pt()/1000.0),
            "mu_eta": cols.mu.Select(lambda x: x.eta()),
            "mu_phi": cols.mu.Select(lambda x: x.phi()),

            "jet_pt":  cols.jets.Select(lambda j: j.pt()/1000.0),
            "jet_eta": cols.jets.Select(lambda j: j.eta()),
            "jet_phi": cols.jets.Select(lambda j: j.phi()),
            "jet_m":   cols.jets.Select(lambda j: j.m()/1000.0),

            "met_px": cols.met.First().mpx()/1000.0,
            "met_py": cols.met.First().mpy()/1000.0,

            "mc_w": cols.mc_w,
        }
    )
)

# Execute the query (only 1 file to keep this light; increase yourself if needed)
result = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="ttbar_mass",
                    Dataset=dataset.Rucio(ds_name),
                    NFiles=1,
                    Query=query,
                )
            ]
        )
    )
)

data = result["ttbar_mass"]

# Build a single lepton per event (exactly one by construction)
has_e = ak.num(data.ele_pt, axis=1) == 1

lep_pt  = ak.where(has_e, ak.firsts(data.ele_pt), ak.firsts(data.mu_pt))
lep_eta = ak.where(has_e, ak.firsts(data.ele_eta), ak.firsts(data.mu_eta))
lep_phi = ak.where(has_e, ak.firsts(data.ele_phi), ak.firsts(data.mu_phi))
# Assign lepton mass based on flavor
lep_mass = ak.where(has_e, ak.ones_like(lep_pt)*0.000511, ak.ones_like(lep_pt)*0.105658)

lep = ak.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "mass": lep_mass}, with_name="Momentum4D")

# MET components (GeV)
met_px = data.met_px
met_py = data.met_py
met2   = met_px**2 + met_py**2

# Solve neutrino pz with W-mass constraint
mW = 80.379  # GeV
a = (mW**2)/2.0 + lep.px*met_px + lep.py*met_py
A = lep.E**2 - lep.pz**2
disc = a**2 - A*met2
disc_clipped = ak.where(disc < 0.0, 0.0, disc)

# Use NumPy ufuncs on Awkward arrays
sqrt_disc = np.sqrt(disc_clipped)

pz1 = (a*lep.pz + lep.E*sqrt_disc) / A
pz2 = (a*lep.pz - lep.E*sqrt_disc) / A
use_first = abs(pz1) < abs(pz2)  # use Python's abs on Awkward arrays
nu_pz = ak.where(use_first, pz1, pz2)
nu_E  = np.sqrt(met2 + nu_pz**2)

nu = ak.zip({"px": met_px, "py": met_py, "pz": nu_pz, "E": nu_E}, with_name="Momentum4D")

# Jets four-vectors
jets = ak.zip(
    {"pt": data.jet_pt, "eta": data.jet_eta, "phi": data.jet_phi, "mass": data.jet_m},
    with_name="Momentum4D",
)

# Sum jets per event (at least two jets per event by selection)
jets_sum = ak.sum(jets, axis=1)

# Reconstructed ttbar 4-vector and mass
ttbar = lep + nu + jets_sum
m_tt = ttbar.mass  # GeV

# Histogram (weighted by MC event weight only)
weights = data.mc_w

h = (
    Hist.new.Reg(50, 0.0, 4000.0, name="mtt", label="Reconstructed $m_{t\\bar{t}}$ [GeV]")
        .Weight()
)
h.fill(mtt=ak.to_numpy(m_tt), weight=ak.to_numpy(weights))

# Plot and save
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()
h.plot(histtype="fill", linewidth=1, edgecolor="gray", label="MC (weighted)")
ax.legend()
ax.set_xlabel("Reconstructed $m_{t\\bar{t}}$ [GeV]")
ax.set_title("Semileptonic $t\\bar{t}$")
fig.savefig("mtt_semilep.png")
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


![a608af5f_gpt-5_mtt_semilep.png](img/a608af5f_gpt-5_mtt_semilep.png)
<details><summary>Usage</summary>
<div style="margin-left: 1em;">


### LLM Usage
| Step | Time (sec) | Prompt Tokens | Completion Tokens | Total Tokens | Cost (USD) |
|---|---|---|---|---|---|
| Run 1 | 187.40 | 14995 | 8341 | 23336 | $0.10 |
| Run 2 | 51.51 | 17500 | 3100 | 20600 | $0.05 |
| **Total** | 238.91 | 32495 | 11441 | 43936 | $0.155 |

### Docker Usage
| Phase | Elapsed Time (seconds) |
|-------|--------------|
| Run 1 | 54.14 |
| Run 2 | 7.65 |
| **Total** | **61.79** |

</div></details>


## CSV

Model,Time,PromptTokens,CompletionTokens,TotalTokens,EstimatedCost,Attempts,CodeTime,Result
gpt-5,238.91,32495,11441,43936,0.155,2,61.79,Success
## Summary
| Model | Time | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost | Attempts | Code Time | Result |
|-----|----|-------------|-----------------|------------|--------------|--------|---------|------|
| gpt-5 | 238.91 | 32495 | 11441 | 43936 | $0.155 | 2 | 61.79 | Success |
