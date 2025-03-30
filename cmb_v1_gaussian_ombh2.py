import numpy as np
from scipy.integrate import quad
from astropy.cosmology import Planck18, Flatw0waCDM
import astropy.units as u
from astropy.constants import c

# Constants
c = 299792.458
T_CMB = 2.7255  # CMB temperature in Kelvin
Omega_gamma_h2 = 2.469e-5 * (T_CMB / 2.7255)**4  # Photon density (Omega_gamma * h^2)

class CMBParams:
    def __init__(self, H0=None, omegam=None, omegabh2=None, omk=None, omnuh2=None, w=None, wa=None, nnu=None):
        self.H0 = H0 if H0 is not None else Planck18.H0.value
        self.omegam = omegam if omegam is not None else Planck18.Om0
        self.omk = omk if omk is not None else 0.0  # Flat universe as default
        self.omnuh2 = omnuh2 if omnuh2 is not None else 0.06  # Default to small neutrino mass contribution
        self.w = w if w is not None else -1.0  # Default to Lambda CDM
        self.wa = wa if wa is not None else 0.0  # Default to w0waCDM with no evolution in w
        self.nnu = nnu if nnu is not None else 3.046  # Default number of neutrino species
        self.h = self.H0 / 100.0
        self.h2 = self.h ** 2
        omega_b_h2_mean = 0.02218
        omega_b_h2_sigma = 0.00055
        self.omegabh2 = omegabh2 if omegabh2 is not None else np.random.normal(omega_b_h2_mean, omega_b_h2_sigma)
        self.Yhe = 0.24  # Default helium fraction
        self.omnu = self.omnuh2 / self.h2  # Neutrino density
        self.ombh2 = self.omegabh2  # Baryon density
        self.omb = self.ombh2 / self.h2  # Baryon density normalized by h^2
        self.omc = self.omegam - self.omnu - self.omb  # Cold dark matter density
        self.omch2 = self.omc * self.h2  # Cold dark matter density scaled by h^2
        self.omdmh2 = self.omch2 + self.omnuh2  # Total matter density (dark matter + neutrino)
        self.omdm = self.omdmh2 / self.h2  # Matter density scaled by h^2
        self.omv = 1 - self.omk - self.omb - self.omdm  # Dark energy density


def set_astropy_cosmology(CMBParams):
    try:
        cosmo = Flatw0waCDM(
            H0=CMBParams.H0,
            Om0=CMBParams.omegam,
            Ob0=CMBParams.omb,
            Tcmb0=T_CMB,
            Neff=CMBParams.nnu,
            m_nu=Planck18.m_nu,
            w0=CMBParams.w,
            wa=CMBParams.wa
        )
        return cosmo
    except Exception as e:
        print(f"Error in setting cosmology: {e}")
        return np.nan

def calculate_z_star(cosmology):
    g_1 = 0.0783 * cosmology.omegabh2 ** -0.238 / (1.0 + 39.5 * cosmology.omegabh2 ** 0.763)
    g_2 = 0.560 / (1.0 + 21.1 * cosmology.omegabh2 ** 1.81)
    z_star = 1048.0 * (1.0 + 0.00124 * cosmology.omegabh2 ** -0.738) * (1.0 + g_1 * (cosmology.omegam*cosmology.h2) ** g_2)
    return z_star

# Function for the sound horizon integrand
#def integrand(a, Omega_b_h2, Omega_gamma_h2, cosmo):
#    E_a = cosmo.efunc(1/a - 1)  # E(a) = H(a) / H0
 #   #E_a = cosmo.efunc(a)
  #  factor = np.sqrt(3 * (1 + (3 * Omega_b_h2 / (4 * Omega_gamma_h2)) * a)**-1)
   # return factor / (a**2 * E_a)

# Function to calculate r_s(z)
#def sound_horizon(z, Omega_b_h2, Omega_gamma_h2, cosmo):
  #  a_start = 0  # Initial value of a
    #a_end = 1 / (1 + z)  # Corresponding a for redshift z
  #  integral_result, _ = quad(integrand, 0, a_end, args=(Omega_b_h2, Omega_gamma_h2, cosmo), epsabs=1e-12, epsrel=1e-12)
   # return (c / H0) * integral_result  # in Mpc

def H_a(a, cosmo):
    z = 1 / a - 1                           # Convert scale factor to redshift
    return cosmo.H(z).value

def integrand_2(a, Omega_b_h2, Omega_gamma_h2, cosmo):
    H_a_val = H_a(a, cosmo)  # Get H(a) using Astropy's H(z)
    factor = np.sqrt(1 + (3 * Omega_b_h2 / (4 * Omega_gamma_h2)) * a)
    return 1 / (a**2 * H_a_val * factor *np.sqrt(3))

def sound_horizon_2(z, Omega_b_h2, Omega_gamma_h2, cosmo):
    #  comoving sound horizon
    a_end = 1 / (1 + z)  # Corresponding a for redshift z
    integral_result, _ = quad(integrand_2, 0, a_end, args=(Omega_b_h2, Omega_gamma_h2, cosmo), epsabs=1e-12, epsrel=1e-12)
    return c * integral_result

def _la(z_star,rs,cosmo):

    la = (1 + z_star) * np.pi * cosmo.angular_diameter_distance(z_star) / rs
    return la.to(u.Mpc).value

def _r_zs(z_star,cosmo):
    #cmb shift
    r_zs = (1+z_star)*cosmo.angular_diameter_distance(z_star)*np.sqrt(cosmo.Om0*cosmo.H0**2)/c
    return r_zs.value


def la_rzs_from_cosmoparams(H0=None, omegam=None, omegabh2=None, omk=None, omnuh2=None, w=None, wa=None, nnu=None):
    cosmology = CMBParams(H0=H0, omegam=omegam, omegabh2=omegabh2, omk=omk, omnuh2=omnuh2, w=w, wa=wa, nnu=nnu)
    astropycosmo = set_astropy_cosmology(cosmology)
    if astropycosmo is np.nan:
        return np.nan, np.nan
    z_star = calculate_z_star(cosmology)

    rs_zstar = sound_horizon_2(z= z_star, Omega_b_h2=cosmology.omegabh2, Omega_gamma_h2=Omega_gamma_h2, cosmo=astropycosmo)

    la = _la(z_star=z_star,rs= rs_zstar, cosmo=astropycosmo)
    r = _r_zs(z_star=z_star,cosmo=astropycosmo)

    return la, r

#z_star = calculate_z_star(CMBParams())
#rs_z_star = sound_horizon(z_star, Omega_b_h2, Omega_gamma_h2, Planck18)
#rs_z_star2 = sound_horizon_2(z_star, Omega_b_h2, Omega_gamma_h2, Planck18)
#la1 = la(z_star,rs_z_star,Planck18)
#la2 = _la(z_star,rs_z_star2,Planck18)
#r_zs = _r_zs(z_star,Planck18)
#la3= la(1090.51,146.8,Planck18)

#print(f"Sound horizon at z_star = {z_star}: r_s(z_star) = {rs_z_star:.4f} Mpc, la = {la1:.4f}")
#print(f"Sound horizon at z_star = {z_star}: r_s(z_star) = {rs_z_star2:.4f} Mpc, la = {la2:.4f}, r_zs = {r_zs:.4f}")
#print(f"Sound horizon at z_star = {1090.51}: r_s(z_star) = {146.8:.4f} Mpc, la = {la3:.4f}")

