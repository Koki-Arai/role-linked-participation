# Role-Linked Participation and Cross-Deterrence in Asymmetric Cartels

Replication and verification materials for the manuscript:

> **Koki Arai, “Role-Linked Participation and Cross-Deterrence in Asymmetric Cartels.”**

This repository contains the deterministic Python code used to verify the numerical illustrations, comparative statics, equilibrium calculations, and robustness checks reported in manuscript version **v6**.

## What this repository reproduces

The script `verification_IJIO_v6.py` checks or reproduces the following results from the manuscript:

1. **Recognition-wedge identity and bottleneck amplification**  
   Verifies the relationship between the non-bottleneck and bottleneck recognition wedges.

2. **Verification precision and curvature**  
   Reproduces the convex / affine / concave cases for the effect of more informative verification on core-firm entry, including the reversal examples.

3. **Cross-deterrence and the burden-timing ratio**  
   Reproduces the equilibrium entry probabilities, the recognition-feedback multiplier, the effect of supporter-specific burdens, and the exact ratio between entry-stage and success-stage burdens.

4. **Multiplicity of entry equilibria**  
   Uses both best-response iteration and a sign scan of the fixed-point equation to identify stable and unstable equilibria. The sign scan is necessary because unstable fixed points are not recovered by simple iteration.

5. **Operational-transparency geometry**  
   Reproduces the turning point of the core firm's continuation margin and the regime intervals used in the high-low-high leverage comparison.

6. **Tipping / supporter-exit threshold**  
   Computes the threshold burden at which active supporter entry disappears and verifies the interior-fold condition `M(p†) = 1`.

7. **Eliminability by regime**  
   Verifies that supporter-directed enforcement can eliminate the arrangement in the bottleneck regime, while in the non-bottleneck regime the core firm can remain active after the supporter exits.

8. **General supporter entry-cost distribution**  
   Checks that the interior-fold characterization `M(p†) = 1` survives beyond the uniform supporter-cost benchmark, using a Beta distribution as a numerical example.

9. **Non-affine recognition-induced exposure**  
   Illustrates the local recognition wedge when the core firm's exposure is nonlinear in its posterior.

## Repository contents

- `verification_IJIO_v6.py` — main verification script.
- `requirements.txt` — Python dependencies used for the final verification run.
- `verification_output_v6.txt` — console output from a reference run of the script.
- `CITATION.cff` — citation metadata for GitHub and reference managers.
- `LICENSE` — MIT license for the code in this repository.
- `.gitignore` — standard exclusions for local Python and editor files.

No confidential or proprietary data are used. The manuscript is theoretical, and all reported numerical illustrations are generated from parameter values stated in the paper.

## Requirements

The final pre-submission verification was run with:

- Python 3.13.5
- NumPy 2.3.5
- SciPy 1.17.0
- mpmath 1.3.0

Install the tested package versions with:

```bash
python -m pip install -r requirements.txt
```

A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell
python -m pip install -r requirements.txt
```

## Running the verification

From the repository root, run:

```bash
python verification_IJIO_v6.py
```

To save a fresh copy of the console output:

```bash
python verification_IJIO_v6.py > verification_output_v6.txt
```

The script requires no command-line arguments and does not download or read external data.

## Interpreting the output

The output is organized by the proposition, corollary, remark, or table being checked. The main quantities include:

- `Theta_N`, `Theta_B` — recognition wedges by regime;
- `p1*` — supporter equilibrium entry probability;
- `Q1` — core-firm entry probability conditional on actual support;
- `M` — recognition-feedback multiplier;
- `dR/dF1E` — response of unconditional core entry to an entry-stage supporter burden;
- `F1E*` — threshold burden associated with supporter exit / tipping;
- `tau*` — turning point in the core firm's stand-alone continuation margin.

Small differences in the final printed decimal place may arise across numerical-library or platform versions. The qualitative conclusions and reported manuscript values should be unchanged.

## Relation to the manuscript

The repository corresponds to **manuscript version v6**. In particular, it reflects the version in which:

- entry-stage activation rents are separated from repeated-game continuation values;
- the tipping condition is written for a general supporter entry-cost distribution;
- the role of non-affine recognition-induced exposure is expressed through a local recognition wedge;
- supporter eliminability is distinguished from supporter exit in the non-bottleneck regime.

If the manuscript is revised after submission, the repository should be tagged or released with a matching version number and the README should be updated accordingly.

## Reproducibility statement

All numerical results in the manuscript are deterministic functions of parameters reported in the paper. No random seed is required for the main verification script. The code is intended to provide an auditable check of the paper's calculations rather than an empirical replication package.

## Citation

If you use the code, please cite the accompanying manuscript and this repository. GitHub can read the citation metadata from `CITATION.cff`.

Repository: <https://github.com/Koki-Arai/role-linked-participation>

## License

The verification code is released under the MIT License. See `LICENSE` for details.

## Contact

**Koki Arai**  
Faculty of Business Studies, Kyoritsu Women's University  
ORCID: <https://orcid.org/0000-0002-6907-4046>
