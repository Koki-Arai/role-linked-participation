# Role-Linked Participation in Cartels: Intermediaries, Pivotality, and the Timing of Enforcement

Replication and verification material for the manuscript:

> **Koki Arai, "Role-Linked Participation in Cartels: Intermediaries, Pivotality, and the Timing of Enforcement."**

This repository contains the deterministic Python code used to verify the theorem, propositions, comparative statics, thresholds, and numerical illustrations reported in manuscript version **v11**.

The manuscript is theoretical. No confidential or proprietary data are used, and the enforcement orders discussed in Section 3 are published in full by the Japan Fair Trade Commission and are identified there by case number and date.

## What this repository reproduces

The script `verification_RIO_v11.py` checks or reproduces the following results.

1. **Recognition-wedge identity and bottleneck amplification** (Proposition 1). Verifies the relationship between the non-bottleneck and bottleneck recognition wedges.

2. **Verification precision and curvature** (Proposition A.1, Appendix A). Reproduces the convex / affine / concave cases for the effect of more informative verification on core-firm entry, including the reversal examples and the strictly concave counterexample to sorting.

3. **Cross-deterrence and the burden-timing ratio** (Proposition 2, Table 2). Reproduces the equilibrium entry probabilities, the recognition-feedback multiplier, the effect of supporter-specific burdens, and the ratio between entry-stage and success-stage burdens.

4. **Multiplicity of entry equilibria** (Lemma 1). Uses both best-response iteration and a sign scan of the fixed-point equation to identify stable and unstable equilibria. The sign scan is necessary because unstable fixed points are not recovered by simple iteration.

5. **Operational-transparency geometry** (Propositions 3 and 4). Reproduces the turning point of the core firm's stand-alone continuation margin and the regime intervals used in the high–low–high leverage comparison.

6. **Tipping thresholds** (Proposition 5, Lemma B.1). Computes the threshold burden at which active supporter entry disappears, verifies the interior-fold condition `M(p†) = 1`, and locates the case (i) / case (ii) boundary by root-finding on `Φ′` rather than by a grid scan.

7. **Eliminability by regime** (Corollary 1). Verifies that supporter-directed enforcement eliminates the arrangement in the bottleneck regime, while in the non-bottleneck regime unconditional core entry settles at `A₂⁰/ē₂` and stays there at any supporter burden.

8. **General supporter entry-cost distribution.** Checks that the interior-fold characterisation `M(p†) = 1` survives beyond the uniform benchmark, using a Beta distribution as a numerical example.

9. **Non-affine recognition-induced exposure** (Remark 2). Illustrates the local recognition wedge when the core firm's exposure is nonlinear in its posterior.

10. **Continuous extinction** (Proposition 5, case (iii)). Confirms that in the exposure-dominant regime `Φ` is strictly decreasing, the supremum is not attained, and supporter entry falls continuously to zero — the case marked `†` in Table 2.

11. **The general timing theorem** (Theorem 1). Checks the incidence formula and the `1/Q` ratio in the general sequential-participation setting of Section 2, using none of the cartel apparatus: a non-uniform cost distribution and four conditional participation functions (increasing convex, increasing concave, decreasing, and constant), together with an arbitrary state-contingent instrument. Also verifies the envelope derivatives at an interior fold.

## Repository contents

- `verification_RIO_v11.py` — verification script.
- `verification_output_v11.txt` — console output from a reference run.
- `requirements.txt` — pinned Python dependencies.
- `CITATION.cff` — citation metadata for GitHub and reference managers.
- `LICENSE` — MIT license for the code in this repository.
- `.gitignore` — exclusions for local Python, LaTeX, and editor files.

## Requirements

- Python 3.12 or later
- NumPy, SciPy, mpmath (pinned versions in `requirements.txt`)

Install the pinned versions with:

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

From the repository root:

```bash
python verification_RIO_v11.py
```

To regenerate the reference output:

```bash
python verification_RIO_v11.py > verification_output_v11.txt
```

The script takes no arguments, reads no external data, and runs in a few seconds.

## Interpreting the output

The output is organised by the result being checked. The main quantities are:

- `Theta_N`, `Theta_B` — recognition wedges by regime;
- `p1*` — supporter equilibrium entry probability;
- `Q1` — core-firm entry probability conditional on actual support;
- `M` — recognition-feedback multiplier; `M(p_dag) = 1` marks the interior fold;
- `dR/dF1E` — response of unconditional core entry to an entry-stage supporter burden;
- `F1E*` — threshold burden at which supporter entry ceases;
- `tau*` — turning point in the core firm's stand-alone continuation margin;
- `dS/dE` and `dC/dE` (block 11) — the ratios predicted by Theorem 1, reported alongside the predicted values and the absolute error.

In block 11 the reported errors are of order 1e-11 or smaller. Elsewhere, small differences in the final printed decimal place may arise across numerical-library or platform versions; the reported manuscript values and the qualitative conclusions are unchanged.

## Relation to earlier versions

Earlier drafts of this project circulated under the title *Role-Linked Participation and Cross-Deterrence in Asymmetric Cartels*, with verification scripts numbered by draft. The model is unchanged from those drafts; the numbering of results changed when the verification-precision material moved to Appendix A and the shape lemma to Appendix B. The mapping is recorded in the docstring of `verification_RIO_v11.py`.

## Reproducibility statement

All numerical results in the manuscript are deterministic functions of parameters reported in the paper. No random seed is required. The code is intended as an auditable check of the paper's calculations rather than an empirical replication package.

## Citation

If you use this code, please cite the accompanying manuscript and this repository. GitHub reads the citation metadata from `CITATION.cff`.

Repository: <https://github.com/Koki-Arai/role-linked-participation>

## License

Released under the MIT License. See `LICENSE` for details.

## Contact

**Koki Arai**
Faculty of Business Studies, Kyoritsu Women's University
ORCID: <https://orcid.org/0000-0002-6907-4046>
