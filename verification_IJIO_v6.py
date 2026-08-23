"""
Verification script for "Role-Linked Participation and Cross-Deterrence in
Asymmetric Cartels", manuscript version v6. Reproduces the numerical results,
comparative-static checks, equilibrium calculations, and robustness checks
reported in the manuscript and used in the final pre-submission review.

Tested with Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, and mpmath 1.3.0.

Orientation, re-stated in every block so it cannot silently flip:
    alpha = f(H|1) > eta = f(H|0),   p1 = supporter entry probability,
    core cutoff  x2^s = a_r + Theta_r * rho_s,   G2 uniform on [0, ebar2] unless stated.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta as Bt
from mpmath import mp, mpf
mp.dps = 40

# ------------------------------------------------------------------ primitives
def rho_HL(p, a, e):
    dH = a*p + e*(1-p); dL = (1-a)*p + (1-e)*(1-p)
    return (a*p/dH if dH > 0 else 0.0), ((1-a)*p/dL if dL > 0 else 0.0)

G2u = lambda x: np.clip(x, 0, 1)                        # affine
G2cx = lambda x: Bt.cdf(np.clip(x, 0, 1), 2, 1)         # convex,  x^2
G2cv = lambda x: Bt.cdf(np.clip(x, 0, 1), 1, 2)         # concave, 1-(1-x)^2
G2vc = lambda x: Bt.cdf(np.clip(x, 0, 1), 1, 5)         # strongly concave

def Q1(p, a, e, aa, Th, G2=G2u):
    rH, rL = rho_HL(p, a, e); return a*G2(aa+Th*rH) + (1-a)*G2(aa+Th*rL)

def R(p, a, e, aa, Th, G2=G2u):
    rH, rL = rho_HL(p, a, e); qH = a*p + e*(1-p)
    return qH*G2(aa+Th*rH) + (1-qH)*G2(aa+Th*rL)

# ---------------------------------------------------------- self-tests first
assert rho_HL(0.0, .8, .2) == (0.0, 0.0)                 # boundary is defined
assert abs(rho_HL(1.0, .8, .2)[0] - 1) < 1e-12
assert abs(rho_HL(0.4, .5, .5)[0] - 0.4) < 1e-12         # uninformative -> rho = p
print("self-tests passed: boundary posteriors, uninformative signal\n")

# ------------------------------------------ (1) Proposition 1: wedge identity
A2_0, Delta, phi = 0.30, 0.40, 0.10
Th_N, Th_B = Delta - phi, A2_0 + Delta - phi
print("Prop 1  Theta_B - Theta_N - A2_0 = %.2e   (identity)" % (Th_B - Th_N - A2_0))

# ------------------------------- (2) Proposition 2(i): three-branch curvature
print("\nProp 2(i)  direct effect of lambda on R, p1 = 0.40 fixed")
for nm, G2 in [('affine', G2u), ('convex', G2cx), ('concave', G2cv), ('v.concave', G2vc)]:
    d = R(.40, .99, .01, A2_0, Th_N, G2) - R(.40, .51, .49, A2_0, Th_N, G2)
    print("   G2=%-10s dR = %+0.6f" % (nm, d))
print("   sign is independent of sign(Theta):  Theta = -0.15, a_N = A2_0 = 0.30")
for nm, G2 in [('convex', G2cx), ('concave', G2cv)]:
    d = R(.40, .99, .01, A2_0, -0.15, G2) - R(.40, .51, .49, A2_0, -0.15, G2)
    print("   G2=%-10s dR = %+0.6f" % (nm, d))

# --------------------------- (3) Proposition 2(ii): the sorting condition binds
print("\nProp 2(ii)  min (rho h)'' versus monotonicity of Q1 in lambda")
lams = np.linspace(0, 1, 201); r = np.linspace(1e-6, 1-1e-6, 2001)
for nm, G2 in [('affine', G2u), ('convex', G2cx), ('concave', G2cv), ('v.concave', G2vc)]:
    curv = np.min(np.diff(r*G2(A2_0+Th_N*r), 2))/(r[1]-r[0])**2
    bad = [p for p in (.05,.2,.4,.6,.8,.95)
           if min(np.diff([Q1(p, .5+.49*l, .5-.49*l, A2_0, Th_N, G2) for l in lams])) < -1e-12]
    print("   G2=%-10s min(rho h)'' = %+8.4f   monotone: %s" %
          (nm, curv, 'YES' if not bad else 'NO at p=%s' % bad))
print("\nProp 2(iii) Theta<0, affine G2, a_N=A2_0: Q1 %.6f -> %.6f (falls)" %
      (Q1(.4, .5, .5, A2_0, -.15), Q1(.4, .99, .01, A2_0, -.15)))

# --------------------- (4) Proposition 3 and Table 1: regimes, ratio, leverage
ebar1, F1E0, kap, F1S0, V1 = 0.70, 0.02, 0.02, 0.05, 1.00
a, e, g1 = 0.8, 0.2, 1/0.70
T = lambda p, A1, FE, aa, Th: min(max((A1*Q1(p,a,e,aa,Th)-FE-kap)/ebar1, 0.0), 1.0)
def fps(A1, FE, aa, Th, starts=(0,.02,.05,.1,.25,.5,.75,.95,1.)):
    out = []
    for p0 in starts:
        p = p0
        for _ in range(400000):
            pn = T(p, A1, FE, aa, Th)
            if abs(pn-p) < 1e-15: break
            p = pn
        out.append(round(p, 9))
    return sorted(set(out))

print("\nTable 1 / Prop 3")
# NOTE: a_N = A2_0 in BOTH regime-N rows.  An earlier draft used a_N = 0.60 in the
# exposure row, which is inconsistent with the calibration A2_0 = 0.30 stated in the
# paper; that produced the wrong Table 1 row 2 and the wrong numbers in Remark 3.
for lab, aa, Th in [("N reliance  ", A2_0, 0.30), ("N exposure  ", A2_0, -0.15),
                    ("B bottleneck", 0.00, 0.60)]:
    h = 1e-7; pf = lambda FE, FS: fps(V1-FS, FE, aa, Th, starts=(0.999,))[-1]
    p = pf(F1E0, F1S0)
    dE = (pf(F1E0+h, F1S0)-pf(F1E0-h, F1S0))/(2*h)
    dS = (pf(F1E0, F1S0+h)-pf(F1E0, F1S0-h))/(2*h)
    q = Q1(p, a, e, aa, Th)
    M = g1*(V1-F1S0)*(Q1(p+h,a,e,aa,Th)-Q1(p-h,a,e,aa,Th))/(2*h)
    print("   %s Theta=%+0.2f p1*=%.4f Q1=%.4f M=%+0.3f dR/dF1E=%+0.3f L=%.3f  "
          "ratio dS/dE=%.4f (Q1=%.4f)" % (lab, Th, p, q, M, Th*dE,
          abs(Th)*g1/(1-M), dS/dE, q))
# Best-response iteration CANNOT find repelling fixed points.  Scan T(p)-p instead.
def all_fps(A1, FE, aa, Th, n=400001):
    f = lambda p: T(p, A1, FE, aa, Th) - p
    xs = np.linspace(1e-9, 1-1e-9, n); v = np.array([f(x) for x in xs]); out = []
    for i in range(n-1):
        if v[i] == 0 or v[i]*v[i+1] < 0:
            r = brentq(f, xs[i], xs[i+1])
            d = (T(r+1e-7,A1,FE,aa,Th)-T(r-1e-7,A1,FE,aa,Th))/2e-7
            out.append((round(r,6), 'stable' if d < 1 else 'unstable', round(d,3)))
    return out
for lab, aa, Th in [("N", A2_0, .30), ("B", 0.0, .60)]:
    it = fps(V1-F1S0, F1E0, aa, Th)
    sc = all_fps(V1-F1S0, F1E0, aa, Th)
    print("   regime %s: iteration finds %s; sign-scan finds interior %s" % (lab, it, sc))
print("   (the unstable point at p=0.0422 in regime B is invisible to iteration)")

# ------------------- (5) Proposition 6: tipping thresholds in both regimes
print("\nProp 6  Phi_r(p) = A1*Q1(p) - ebar1*p ;  F1E* = max Phi_r - kappa1")
A1v = V1 - F1S0
def Phi(p, aa, Th): return A1v*Q1(p, a, e, aa, Th) - ebar1*p
gr = np.linspace(0, 1, 400001)
for lab, aa, Th in [("N (a=0.30, Theta=0.30)", A2_0, 0.30), ("B (a=0.00, Theta=0.60)", 0.0, 0.60)]:
    vv = np.array([Phi(x, aa, Th) for x in gr]); i = int(np.argmax(vv))
    pdag, Fst = gr[i], vv[i] - kap
    Mdag = g1*A1v*(Q1(pdag+1e-7,a,e,aa,Th)-Q1(pdag-1e-7,a,e,aa,Th))/2e-7
    T0lim = A1v*Q1(0,a,e,aa,Th) - kap          # F1E above which p=0 is an equilibrium
    print("   %s  Phi(0)=%.4f  p_dag=%.6f  M(p_dag)=%.4f  F1E*=%.6f  (p=0 equilibrium once F1E>=%.4f)"
          % (lab, vv[0], pdag, Mdag, Fst, max(T0lim, 0.0)))
FN = max(Phi(x, A2_0, 0.30) for x in gr) - kap
FB = max(Phi(x, 0.0, 0.60) for x in gr) - kap
print("   ratio F1E*(B) / F1E*(N) = %.4f" % (FB/FN))
# envelope derivatives at the bottleneck fold
vv = np.array([Phi(x,0.0,0.60) for x in gr]); pdag = gr[int(np.argmax(vv))]
def Erho(p): 
    rH,rL = rho_HL(p,a,e); return a*rH+(1-a)*rL
print("   envelope: dF*/dkappa1=%+0.4f  dF*/dF1S=%+0.4f  dF*/debar1=%+0.4f  dF*/dTheta=%+0.4f"
      % (-1.0, -Q1(pdag,a,e,0.0,0.60), -pdag, A1v*Erho(pdag)))

# ---------------- (5b) Lemma 2: shape of Phi, and the boundary slope
print("\nLemma 2  alpha*rhoH' + (1-alpha)*rhoL' is convex; Phi'(1) = A1*Theta/ebar2 - ebar1")
for al, et in [(0.8,0.2), (0.55,0.01), (0.99,0.5), (0.6,0.45)]:
    x = np.linspace(1e-4, 1-1e-4, 50001)
    w = al*(al*et/(al*x+et*(1-x))**2) + (1-al)*((1-al)*(1-et)/((1-al)*x+(1-et)*(1-x))**2)
    print("   alpha=%.2f eta=%.2f: min 2nd diff = %.2e (convex), w(1)=%.6f (should be 1)"
          % (al, et, np.diff(w, 2).min(), w[-1]))
print("   => Phi' is U-shaped, so Phi has at most one interior max then one interior min;")
print("      single-peakedness is FALSE in general and max Phi = max{Phi(p_dag), Phi(1)}.")
# Locate p_dag by root-finding on Phi', not by scanning a grid: near the
# case (i)/(ii) boundary the two candidate maxima differ in the third decimal.
def phi_prime(pv, eb1v, Thv):
    h2 = 1e-8
    return (A1v*Q1(pv+h2, a, e, 0.0, Thv) - A1v*Q1(pv-h2, a, e, 0.0, Thv))/(2*h2) - eb1v
for eb1v, Thv in [(0.70, 0.60), (0.20, 0.60), (0.10, 0.60)]:
    Phiv = lambda pv: A1v*Q1(pv, a, e, 0.0, Thv) - eb1v*pv
    xs2 = np.linspace(1e-6, 1-1e-6, 4001); dv = np.array([phi_prime(x, eb1v, Thv) for x in xs2])
    pdag2 = None
    for j in range(len(xs2)-1):
        if dv[j] > 0 >= dv[j+1]:
            pdag2 = brentq(lambda pv: phi_prime(pv, eb1v, Thv), xs2[j], xs2[j+1]); break
    inter = Phiv(pdag2) if pdag2 is not None else None
    print("   ebar1=%.2f Theta=%.2f: Phi'(1)=%+0.4f  Phi(p_dag)=%s  Phi(1)=%+0.5f  -> case %s"
          % (eb1v, Thv, A1v*Thv - eb1v,
             ("%+0.5f @ p=%.4f" % (inter, pdag2)) if pdag2 is not None else "      none      ",
             Phiv(1.0), "(i) interior" if (inter is not None and inter >= Phiv(1.0)) else "(ii) corner"))

# --------------------------------- (6) Propositions 4-5 and Figure 1 geometry
beta, a2, c2, d2 = 0.80, 0.60, 0.10, 0.50
pCD, pCP, gamma = -0.50, 0.50, 0.60
Gam = lambda t: pCD + (c2-d2)*t + beta/(1-beta)*(1-np.exp(-a2*t))*pCP
tstar = (1/a2)*np.log(beta*a2*pCP/((1-beta)*(d2-c2)))
tlo, thi = brentq(Gam, 1e-9, tstar), brentq(Gam, tstar, 60)
tdead = brentq(lambda t: Gam(t)+gamma, tstar, 80)
grid = np.linspace(1e-6, 3*tstar, 400001); vals = Gam(grid)
print("\nProp 4/5  tau* analytic = %.6f, numeric argmax = %.6f, Gamma'' max = %.2e"
      % (tstar, grid[int(np.argmax(vals))], np.max(np.diff(vals, 2))))
print("   Gamma2_0(tau*) = %+0.4f;  N = [%.4f, %.4f];  V = [0, %.4f]"
      % (Gam(tstar), tlo, thi, tdead))


# =============== (7) Corollary 2: eliminability dichotomy =====================
print("\nCor 2  what a supporter-specific burden buys, by regime")
def R_of(FE, aa, Th):
    p = fps(V1-F1S0, FE, aa, Th, starts=(0.999,))[-1]
    return p, (aa + Th*p)          # ebar2 = 1
print("   %8s | %-22s | %-22s" % ("F1E", "regime N  (p1*, R)", "regime B  (p1*, R)"))
for FE in [0.02, 0.10, 0.2650, 0.2704, 0.30, 1.0, 10.0]:
    pN, RN = R_of(FE, A2_0, 0.30); pB, RB = R_of(FE, 0.0, 0.60)
    print("   %8.4f | (%7.5f, %6.4f)       | (%7.5f, %6.4f)" % (FE, pN, RN, pB, RB))
print("   => regime N: R never falls below a_N/ebar2 = %.4f, at ANY supporter burden." % (A2_0))
print("      regime B: R -> 0 once F1E exceeds F1E* = 0.0752.")

# =============== (8) Proposition 6 under a NON-uniform G1 =====================
from scipy.stats import beta as _Bt
print("\nProp 6(i)  the fold is M(p_dag)=1 for a general G1, not only the uniform one")
for nm, G1inv, g1f in [("uniform[0,0.7]", lambda q: 0.70*q, lambda x: 1/0.70),
                       ("Beta(1,3) on [0,1]", lambda q: _Bt.ppf(q, 1, 3), lambda x: _Bt.pdf(min(max(x,0),1), 1, 3))]:
    Phig = lambda q: A1v*Q1(q, a, e, 0.0, 0.60) - G1inv(q)
    xs3 = np.linspace(1e-4, 1-1e-4, 200001); vv3 = np.array([Phig(x) for x in xs3])
    j = int(np.argmax(vv3)); pdg = xs3[j]
    Mg = g1f(G1inv(pdg))*A1v*(Q1(pdg+1e-6,a,e,0.0,0.60)-Q1(pdg-1e-6,a,e,0.0,0.60))/2e-6
    print("   G1=%-20s p_dag=%.5f  M(p_dag)=%.5f  F1E*=%.5f" % (nm, pdg, Mg, vv3[j]-kap))

# =============== (9) Remark 3: local wedge under non-affine F2 ===============
print("\nRemark 3  local wedge Theta_N(rho) = Delta - F2'(rho)")
for nm, F2p in [("affine, phi=0.10", lambda r: 0.10),
                ("convex, F2'=0.3*rho", lambda r: 0.3*r),
                ("concave, F2'=0.3*sqrt(rho)", lambda r: 0.3*np.sqrt(r))]:
    print("   %-28s Theta_N(rho) at rho=0,.5,1: %s" %
          (nm, [round(0.40-F2p(r), 4) for r in (0.0, 0.5, 1.0)]))
print("   => affine F2 is what makes the local wedge constant and the comparative statics global.")
