"""
polaron_tmatrix_example.py

Example numerical implementation (simple, pedagogical) of:
 - Π(Ω,Q) bubble integral
 - T(Ω,Q) with a simple renormalization by scattering length a (non-relativistic matching)
 - Σ(ω,p) for an impurity (optionally heavy/static approximation)
 - Root finding for polaron energy E_p (p=0)

Notes/Limitations:
 - This script is pedagogical. The renormalization used (m_r/(2π a)) assumes non-relativistic matching.
 - Integrals are evaluated with simple nested quadrature and a finite k_max; increase k_max and tolerance for production runs.
 - The "static impurity" limit is implemented by setting M very large (or using flag static=True).
"""

import numpy as np
from scipy import integrate, optimize
import cmath
import math

# ---------- Physical parameters (change as desired) ----------
m = 1.0          # mass of background fermions (units where ħ=c=1)
M = 1e6          # mass of impurity -> large = static approx
mu = 1.2         # chemical potential of background (choose mu>m to have a Fermi sea)
a_scatt = -1.0   # scattering length (negative = attractive)
eta = 1e-6       # small imaginary regulator for retarded functions
k_max = 10.0     # UV cutoff for k integrals (increase for convergence)
m_r = (m * M) / (m + M)  # reduced mass (approx m for static impurity)
# For static impurity, set M >> m so m_r ≈ m

# ---------- Derived quantities ----------
def kF_from_mu(mu, m):
    if mu <= m:
        return 0.0
    return math.sqrt(mu*mu - m*m)

kF = kF_from_mu(mu, m)
print(f"Parameters: m={m}, M={M}, mu={mu}, kF={kF:.6f}, a={a_scatt}")

# ---------- Single-particle energies ----------
def eps_psi(k):
    return math.sqrt(k*k + m*m)

def xi_psi(k):
    return eps_psi(k) - mu

def eps_chi(k):
    return math.sqrt(k*k + M*M)

def xi_chi_k_of_Q_minus_k(Q, k, cos_theta):
    # |Q - k| = sqrt(Q^2 + k^2 - 2 Q k cosθ)
    Qmag = Q
    kmag = k
    arg = Qmag*Qmag + kmag*kmag - 2.0*Qmag*kmag*cos_theta
    if arg < 0: arg = 0.0
    return math.sqrt(arg + M*M) - 0.0  # we set mu_chi reference 0 for impurity

# ---------- Occupation (T=0 Fermi sea) ----------
def nF_psi(k):
    return 1.0 if xi_psi(k) < 0 else 0.0

# ---------- Bubble Π(Ω,Q) ----------
def Pi_of_Omega_Q(Omega, Q, k_max=k_max, eta=eta):
    # integrate over k in [0,k_max] and cosθ in [-1,1]
    def integrand_cos(cos_theta, k):
        xi_k = xi_psi(k)
        xi_Qk = xi_chi_k_of_Q_minus_k(Q, k, cos_theta)  # mu_chi taken as 0 reference
        numer = 1.0 - nF_psi(k)  # for dilute impurity nF_chi≈0
        denom = (Omega - xi_k - xi_Qk) + 1j*eta
        return numer / denom

    def integrand_k(k):
        # integrate cosθ
        real_part = integrate.quad(lambda ct: np.real(integrand_cos(ct, k)), -1.0, 1.0, epsabs=1e-6, epsrel=1e-6)[0]
        imag_part = integrate.quad(lambda ct: np.imag(integrand_cos(ct, k)), -1.0, 1.0, epsabs=1e-6, epsrel=1e-6)[0]
        return k*k * (real_part + 1j*imag_part)

    # radial integral
    real_k = integrate.quad(lambda kk: np.real(integrand_k(kk)), 0.0, k_max, epsabs=1e-6, epsrel=1e-6)[0]
    imag_k = integrate.quad(lambda kk: np.imag(integrand_k(kk)), 0.0, k_max, epsabs=1e-6, epsrel=1e-6)[0]
    Pi = (1.0/(2.0*math.pi**2)) * (real_k + 1j*imag_k)
    return Pi

# ---------- Regularization term (vacuum subtraction) ----------
def vacuum_subtraction_integral(k_max=k_max):
    # ∫ d^3k/(2π)^3 [1/(2 ε_k^rel)]
    # we take ε_k^rel = sqrt(k^2 + m_r^2) as a simple choice for regularization
    def integrand(k):
        return k*k / (2.0 * math.sqrt(k*k + m_r*m_r))
    real = integrate.quad(lambda kk: integrand(kk), 0.0, k_max, epsabs=1e-8, epsrel=1e-8)[0]
    return (1.0/(2.0*math.pi**2)) * real

# ---------- I(Ω,Q) used in T-matrix denominator ----------
def I_of_Omega_Q(Omega, Q, k_max=k_max, eta=eta):
    Pi = Pi_of_Omega_Q(Omega, Q, k_max=k_max, eta=eta)
    vac = vacuum_subtraction_integral(k_max=k_max)
    return Pi + vac

# ---------- T-matrix ----------
def T_of_Omega_Q(Omega, Q, a=a_scatt, k_max=k_max, eta=eta):
    # simple renormalization: 1/T = m_r/(2π a) - I(Omega,Q)
    pref = m_r / (2.0 * math.pi * a)
    I = I_of_Omega_Q(Omega, Q, k_max=k_max, eta=eta)
    denom = pref - I
    return 1.0 / denom

# ---------- Self-energy Σ(ω,p) ----------
def Sigma_of_omega_p(omega, p=0.0, k_max=k_max, eta=eta):
    # integrate over q (occupied states q<kF) and cosθ. For p=0, |p+q|=q
    def integrand_cos(cos_theta, q):
        Qmag = math.sqrt(p*p + q*q + 2.0*p*q*cos_theta)
        Tval = T_of_Omega_Q(omega + xi_psi(q), Qmag, k_max=k_max, eta=eta)
        return Tval

    def integrand_q(q):
        if nF_psi(q) <= 0.0:
            return 0.0  # only occupied states contribute nF
        real_ct = integrate.quad(lambda ct: np.real(integrand_cos(ct, q)), -1.0, 1.0, epsabs=1e-6, epsrel=1e-6)[0]
        imag_ct = integrate.quad(lambda ct: np.imag(integrand_cos(ct, q)), -1.0, 1.0, epsabs=1e-6, epsrel=1e-6)[0]
        return q*q * (real_ct + 1j*imag_ct)

    real_q = integrate.quad(lambda qq: np.real(integrand_q(qq)), 0.0, k_max, epsabs=1e-6, epsrel=1e-6)[0]
    imag_q = integrate.quad(lambda qq: np.imag(integrand_q(qq)), 0.0, k_max, epsabs=1e-6, epsrel=1e-6)[0]
    Sigma = (1.0/(2.0*math.pi**2)) * (real_q + 1j*imag_q)
    return Sigma

# ---------- Polaron energy (root find) ----------
def find_polaron_energy(p=0.0):
    # reference impurity energy (we measure relative to M's rest mass) set to 0 here
    eps_chi_ref = 0.0
    def f(E):
        S = Sigma_of_omega_p(E, p=p)
        return (E - eps_chi_ref - np.real(S))
    # bracket: look for root in [-5, 5] (tune as needed)
    try:
        sol = optimize.root_scalar(f, bracket=[-5.0, 5.0], method='bisect', tol=1e-4)
        if sol.converged:
            return sol.root
        else:
            return None
    except Exception as e:
        print("Root finding failed:", e)
        return None

# ---------- Quick test run ----------
if __name__ == "__main__":
    print("Running a short test (this may take ~tens of seconds depending on k_max and tolerances)...")
    Omega_test = 0.5
    Q_test = 0.0
    Pi_test = Pi_of_Omega_Q(Omega_test, Q_test)
    print(f"Pi({Omega_test}, {Q_test}) = {Pi_test}")

    T_test = T_of_Omega_Q(Omega_test, Q_test)
    print(f"T({Omega_test}, {Q_test}) = {T_test}")

    Sigma_test = Sigma_of_omega_p(0.0)
    print(f"Sigma(0) = {Sigma_test}")

    E_pol = find_polaron_energy(p=0.0)
    print("Estimated polaron energy E_p (p=0):", E_pol)

