from multiprocessing import get_context, Pool
import emcee
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
from astropy.cosmology import Planck18
from scipy.integrate import simpson
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import inv, LinAlgError
import numpy as np
from astropy.cosmology import default_cosmology
import astropy.units as u
from astropy import constants as const
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import gzip
import random

_inv_cov_matrix_sne = np.load("inv_cov_matrix_sne.npy")


def dh_inmpc(z, H0, Om0, w0, wa):
    model = Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)
    # speed of light:
    c = const.c
    c.to(u.km / u.s)
    dh = (c) / model.H(z)
    return dh.to(u.Mpc)  # Mpc


def dm_inmpc(z, H0, Om0, w0, wa):
    model = Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)
    dm = model.comoving_transverse_distance(z)
    return dm


def bao_dv_distance(z, H0, Om0, w0, wa):
    model = Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)
    dm = model.comoving_transverse_distance(z)
    dh = dh_inmpc(z, H0, Om0, w0, wa)  # MPC

    dv = ((dm ** 2) * dh * (z)) ** (1 / 3)
    return dv

def rdrag_desy1(Om0, H0):
    r_d = 147.05 * u.Mpc
    N_eff= Planck18.Neff
    h= H0/100
    omega_b_h2_mean = 0.02218
    omega_b_h2_sigma = 0.00055
    omega_b_h2 = np.random.normal(omega_b_h2_mean, omega_b_h2_sigma)
    omega_m_h2 = Om0 *h**2
    r_drag = r_d * (omega_m_h2 / 0.1432) ** (-0.23) * (N_eff / 3.04) ** (-0.1) * (omega_b_h2 / 0.02236) ** (-0.13)
    return r_drag

def rdrag_desy2():
    r_d = 147.09
    r_d_err = 0.26
    r_drag = np.random.normal(loc=r_d, scale=r_d_err)
    return r_drag* u.Mpc

def bao_dv_dr(z, H0, Om0, w0, wa):
    dv = bao_dv_distance(z, H0, Om0, w0, wa)
    rdrag = rdrag_desy1(Om0, H0)
    #rdrag = rdrag_desy2()
    return dv / rdrag


def bao_fap(z, H0, Om0, w0, wa):
    model = Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)

    dm = model.comoving_transverse_distance(z)
    dh = dh_inmpc(z, H0, Om0, w0, wa)
    fap = dm / dh
    return fap


def sne_distmod(z, H0, Om0, w0, wa, cosmo=None, model_M=None):
    if cosmo is None:
        model = Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)
    else:
        model = cosmo
    if model_M is None:
        model_M = 0.0
    return model.distmod(z).value + model_M


def sne_luminosity_distance(z, H0, Om0, w0, wa, cosmo=None):
    if cosmo is None:
        model = Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)
    else:
        model = cosmo
    return model.luminosity_distance(z).value  # in Mpc


def run_cosmological_distances_multiprocessing(
        n_iterations,
        H0,
        z,
        Om0_range,
        w0_range,
        wa_range,
        lcdm_bao_dv,
        lcdm_bao_fap,
        lcdm_sne_luminosity_distance
):
    indices = [(i, j, k) for i in range(len(w0_range)) for j in range(len(wa_range)) for k in range(len(Om0_range))]
    args = [(i, j, k, H0, z, Om0_range, w0_range, wa_range, lcdm_bao_dv, lcdm_bao_fap, lcdm_sne_luminosity_distance) for
            i, j, k in
            indices]

    with get_context("spawn").Pool() as pool:
        results = pool.starmap(worker_cosmological_distances, args)

    shift_dv = np.zeros((len(w0_range), len(wa_range), len(Om0_range)))
    shift_fap = np.zeros_like(shift_dv)
    shift_des = np.zeros_like(shift_dv)

    for res in results:
        i, j, k, shift_dv_val, shift_fap_val, shift_des_val = res
        shift_dv[i, j, k] = shift_dv_val
        shift_fap[i, j, k] = shift_fap_val
        shift_des[i, j, k] = shift_des_val

    return shift_dv, shift_fap, shift_des


def worker_cosmological_distances(i, j, k, H0, z, Om0_range, w0_range, wa_range, lcdm_bao_dv, lcdm_bao_fap,
                                  lcdm_sne_luminosity_distance):
    w0 = w0_range[i]
    wa = wa_range[j]
    Om0 = Om0_range[k]
    ones = np.ones_like(z)

    dv_dr = bao_dv_dr(z, H0, Om0, w0, wa)
    fap_distance = bao_fap(z, H0, Om0, w0, wa)
    des_luminosity_distance = sne_luminosity_distance(z, H0, Om0, w0, wa)

    ratio_dv = dv_dr / lcdm_bao_dv
    iratio_dv_dimensionless = ratio_dv.unit.is_equivalent(u.dimensionless_unscaled)
    assert iratio_dv_dimensionless, "The ratio of dv_dr should be dimensionless"
    ratio_fap = fap_distance / lcdm_bao_fap
    iratio_fap_dimensionless = ratio_fap.unit.is_equivalent(u.dimensionless_unscaled)
    assert iratio_fap_dimensionless, "The ratio of fap_distance should be dimensionless"
    ratio_des = des_luminosity_distance / lcdm_sne_luminosity_distance

    shift_dv = np.abs(simpson(y=np.abs(ratio_dv - ones), x=z))
    shift_fap = np.abs(simpson(y=np.abs(ratio_fap - ones), x=z))
    shift_des = np.abs(simpson(y=np.abs(ratio_des - ones), x=z))

    return (i, j, k, shift_dv, shift_fap, shift_des)


def log_likelihood(params, z, data_obs, data_err, type, H0 = 70.0 ,cov_matrix_sne=None):
    if type in ['combined_luminosity_distance_lcdm', 'sne_luminosity_distance_lcdm', 'sne_lcdm_distmod',
                'combined_distmod_lcdm']:
        Om0 = params[0]
        M = params[1] if len(params) > 1 else None
        w0 = -1
        wa = 0
    elif type in ['combined_luminosity_distance_lcdm_with_H0', 'sne_luminosity_distance_lcdm_with_H0',
                  'sne_lcdm_distmod_with_H0', 'combined_distmod_lcdm_with_H0']:
        Om0 = params[0]
        H0 = params[1]
        M = params[2] if len(params) > 2 else None
        w0 = -1
        wa = 0
    elif type in ['sne_luminosity_distance_with_H0', 'sne_distmod_with_H0', 'dv_with_H0', 'fap_with_H0',
                  'combined_luminosity_distance_with_H0', 'combined_distmod_with_H0']:
        Om0 = params[0]
        w0 = params[1]
        wa = params[2]
        H0 = params[3]
        M = params[4] if len(params) > 4 else None
    else:
        Om0 = params[0]
        w0 = params[1]
        wa = params[2]
        M = params[3] if len(params) > 3 else None

    if cov_matrix_sne is None:
        inv_cov_matrix_sne = _inv_cov_matrix_sne
    inv_cov_matrix_sne = _inv_cov_matrix_sne

    if M is None:
        M=0.0

    if w0+wa > 0:
        return -np.inf

    if not (0.001 <= Om0 <= 3.0):
        return -np.inf
    if not (-3 <= w0 <= 1):
        return -np.inf
    if not (-10 <= wa <= 5):
        return -np.inf
    if not (50 <= H0 <= 100):
        return -np.inf
    if not (-1.1 <= M <= 1.1):
        return -np.inf
    if type == 'sne_luminosity_distance':
        data_model = sne_luminosity_distance(z, H0=H0, Om0=Om0, w0=w0, wa=wa)
        delta_E = data_obs - data_model
        if cov_matrix_sne is not None:
            chi2 = delta_E @ inv_cov_matrix_sne @ delta_E.T
            #ln_det_cov = np.linalg.slogdet(cov_matrix_sne)[1]
            ln_det_cov = -5654.051096497305
            N_S = len(delta_E)
            ln_L = -0.5 * (chi2 + N_S * np.log(2 * np.pi) + ln_det_cov)
        else:
            chi2 = np.sum((delta_E / data_err) ** 2)
            ln_L = -0.5 * (chi2 + np.sum(np.log(2 * np.pi * data_err ** 2)))
        return ln_L

    elif type == 'sne_luminosity_distance_with_H0':
        data_model = sne_luminosity_distance(z, H0=H0, Om0=Om0, w0=w0, wa=wa)
        delta_E = data_obs - data_model
        if cov_matrix_sne is not None:
            chi2 = delta_E @ inv_cov_matrix_sne @ delta_E.T
            #ln_det_cov = np.linalg.slogdet(cov_matrix_sne)[1]
            ln_det_cov = -5654.051096497305
            N_S = len(delta_E)
            ln_L = -0.5 * (chi2 + N_S * np.log(2 * np.pi) + ln_det_cov)
        else:
            chi2 = np.sum((delta_E / data_err) ** 2)
            ln_L = -0.5 * (chi2 + np.sum(np.log(2 * np.pi * data_err ** 2)))
        return ln_L


    elif type == 'sne_distmod':
        data_model = sne_distmod(z, H0=H0, Om0=Om0, w0=w0, wa=wa,model_M=M)
        delta_E = data_obs - data_model

        if cov_matrix_sne is not None:
            chi2 = delta_E @ inv_cov_matrix_sne @ delta_E.T
            #ln_det_cov = np.linalg.slogdet(cov_matrix_sne)[1]
            ln_det_cov = -5654.051096497305
            N_S = len(delta_E)
            ln_L = -0.5 * (chi2 + N_S * np.log(2 * np.pi) + ln_det_cov)
        else:
            chi2 = np.sum((delta_E / data_err) ** 2)
            ln_L = -0.5 * (chi2 + np.sum(np.log(2 * np.pi * data_err ** 2)))
        return ln_L

    elif type == 'sne_distmod_with_H0':
        data_model = sne_distmod(z, H0=H0, Om0=Om0, w0=w0, wa=wa,model_M=M)
        delta_E = data_obs - data_model

        if cov_matrix_sne is not None:
            chi2 = delta_E @ inv_cov_matrix_sne @ delta_E.T
            #ln_det_cov = np.linalg.slogdet(cov_matrix_sne)[1]
            ln_det_cov = -5654.051096497305
            N_S = len(delta_E)
            ln_L = -0.5 * (chi2 + N_S * np.log(2 * np.pi) + ln_det_cov)
        else:
            chi2 = np.sum((delta_E / data_err) ** 2)
            ln_L = -0.5 * (chi2 + np.sum(np.log(2 * np.pi * data_err ** 2)))
        return ln_L

    elif type == 'dv':
        data_model = bao_dv_dr(z, H0=70.0, Om0=Om0, w0=w0, wa=wa)
        delta_E = data_obs - data_model
        chi2 = np.sum((delta_E / data_err) ** 2)
        ln_L = -0.5 * (chi2 + np.sum(np.log(2 * np.pi * data_err ** 2)))
        return ln_L

    elif type == 'dv_with_H0':
        data_model = bao_dv_dr(z, H0=H0, Om0=Om0, w0=w0, wa=wa)
        delta_E = data_obs - data_model
        chi2 = np.sum((delta_E / data_err) ** 2)
        ln_L = -0.5 * (chi2 + np.sum(np.log(2 * np.pi * data_err ** 2)))
        return ln_L

    elif type == 'fap':
        data_model = bao_fap(z, H0=70.0, Om0=Om0, w0=w0, wa=wa)
        delta_E = data_obs - data_model
        chi2 = np.sum((delta_E / data_err) ** 2)
        ln_L = -0.5 * (chi2 + np.sum(np.log(2 * np.pi * data_err ** 2)))
        return ln_L

    elif type == 'fap_with_H0':
        data_model = bao_fap(z, H0=H0, Om0=Om0, w0=w0, wa=wa)
        delta_E = data_obs - data_model
        chi2 = np.sum((delta_E / data_err) ** 2)
        ln_L = -0.5 * (chi2 + np.sum(np.log(2 * np.pi * data_err ** 2)))
        return ln_L

    elif type == 'sne_luminosity_distance_lcdm':
        data_model = sne_luminosity_distance(z, H0=70.0, Om0=Om0, w0=-1, wa=0)
        delta_E = data_obs - data_model

        if cov_matrix_sne is not None:
            chi2 = delta_E @ inv_cov_matrix_sne @ delta_E.T
            #ln_det_cov = np.linalg.slogdet(cov_matrix_sne)[1]
            ln_det_cov = -5654.051096497305
            N_S = len(delta_E)
            ln_L = -0.5 * (chi2 + N_S * np.log(2 * np.pi) + ln_det_cov)
        else:
            chi2 = np.sum((delta_E / data_err) ** 2)
            ln_L = -0.5 * (chi2 + np.sum(np.log(2 * np.pi * data_err ** 2)))
        return ln_L

    elif type == 'sne_luminosity_distance_lcdm_with_H0':
        data_model = sne_luminosity_distance(z, H0=H0, Om0=Om0, w0=-1, wa=0)
        delta_E = data_obs - data_model
        if cov_matrix_sne is not None:
            chi2 = delta_E @ inv_cov_matrix_sne @ delta_E.T
            #ln_det_cov = np.linalg.slogdet(cov_matrix_sne)[1]
            ln_det_cov = -5654.051096497305
            N_S = len(delta_E)
            ln_L = -0.5 * (chi2 + N_S * np.log(2 * np.pi) + ln_det_cov)
        else:
            chi2 = np.sum((delta_E / data_err) ** 2)
            ln_L = -0.5 * (chi2 + np.sum(np.log(2 * np.pi * data_err ** 2)))
        return ln_L

    elif type == 'sne_lcdm_distmod':
        data_model = sne_distmod(z, H0=70.0, Om0=Om0, w0=-1, wa=0,model_M=M)
        delta_E = data_obs - data_model
        if cov_matrix_sne is not None:
            chi2 = delta_E @ inv_cov_matrix_sne @ delta_E.T
            #ln_det_cov = np.linalg.slogdet(cov_matrix_sne)[1]
            ln_det_cov = -5654.051096497305
            N_S = len(delta_E)
            ln_L = -0.5 * (chi2 + N_S * np.log(2 * np.pi) + ln_det_cov)
        else:
            chi2 = np.sum((delta_E / data_err) ** 2)
            ln_L = -0.5 * (chi2 + np.sum(np.log(2 * np.pi * data_err ** 2)))
        return ln_L

    elif type == 'sne_lcdm_distmod_with_H0':
        data_model = sne_distmod(z, H0=H0, Om0=Om0, w0=-1, wa=0,model_M=M)
        delta_E = data_obs - data_model
        if cov_matrix_sne is not None:
            chi2 = delta_E @ inv_cov_matrix_sne @ delta_E.T
            #ln_det_cov = np.linalg.slogdet(cov_matrix_sne)[1]
            ln_det_cov = -5654.051096497305
            N_S = len(delta_E)
            ln_L = -0.5 * (chi2 + N_S * np.log(2 * np.pi) + ln_det_cov)
        else:
            chi2 = np.sum((delta_E / data_err) ** 2)
            ln_L = -0.5 * (chi2 + np.sum(np.log(2 * np.pi * data_err ** 2)))
        return ln_L

    elif type.startswith('combined'):
        ln_L_total = 0

        if 'sne' in data_obs:
            z_sne = z['sne']
            if 'distmod' in type:
                data_model_sne = sne_distmod(z_sne, H0=H0 if 'with_H0' in type else 70.0, Om0=Om0, w0=w0, wa=wa,model_M=M)
            else:
                data_model_sne = sne_luminosity_distance(z_sne, H0=H0 if 'with_H0' in type else 70.0, Om0=Om0, w0=w0,
                                                         wa=wa)
            delta_E_sne = data_obs['sne'] - data_model_sne

            if cov_matrix_sne is not None:
                chi2_sne = delta_E_sne.T @ np.linalg.solve(cov_matrix_sne, delta_E_sne)
                #ln_det_cov_sne = np.linalg.slogdet(cov_matrix_sne)[1]
                ln_det_cov_sne = -5654.051096497305
                N_S = len(delta_E_sne)
                ln_L_sne = -0.5 * (chi2_sne + N_S * np.log(2 * np.pi) + ln_det_cov_sne)
            else:
                chi2_sne = np.sum((delta_E_sne / data_err['sne']) ** 2)
                ln_L_sne = -0.5 * (chi2_sne + np.sum(np.log(2 * np.pi * data_err['sne'] ** 2)))
            ln_L_total += ln_L_sne

        if 'dv' in data_obs:
            z_dv = z['dv']
            data_model_dv = bao_dv_dr(z_dv, H0=H0 if 'with_H0' in type else 70.0, Om0=Om0, w0=w0, wa=wa)
            delta_E_dv = data_obs['dv'] - data_model_dv
            chi2_dv = np.sum((delta_E_dv / data_err['dv']) ** 2)
            ln_L_dv = -0.5 * (chi2_dv + np.sum(np.log(2 * np.pi * data_err['dv'] ** 2)))
            ln_L_total += ln_L_dv
        if 'fap' in data_obs:
            z_fap = z['fap']
            data_model_fap = bao_fap(z_fap, H0=H0 if 'with_H0' in type else 70.0, Om0=Om0, w0=w0, wa=wa)
            delta_E_fap = data_obs['fap'] - data_model_fap
            chi2_fap = np.sum((delta_E_fap / data_err['fap']) ** 2)
            ln_L_fap = -0.5 * (chi2_fap + np.sum(np.log(2 * np.pi * data_err['fap'] ** 2)))
            ln_L_total += ln_L_fap

        return ln_L_total

    else:
        print(f'{type} not recognized')
        raise ValueError

def run_mcmc_with_cov(z, data_obs, data_err, type, initial_guess, ndim=3, nwalkers=50, nsteps=5000, nburn=100, nthin=15):
    pos = [initial_guess + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(z, data_obs, data_err, type), pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    samples = sampler.get_chain(discard=nburn, thin=nthin, flat=True)
    return samples


def type_to_lcdm(type):
    if type == 'sne_luminosity_distance':
        _type = 'sne_luminosity_distance_lcdm'
    elif type == 'sne_distmod':
        _type = 'sne_lcdm_distmod'
    elif type == 'combined_luminosity_distance':
        _type = 'combined_luminosity_distance_lcdm'
    elif type == 'combined_distmod':
        _type = 'combined_distmod_lcdm'
    elif type == 'sne_luminosity_distance_with_H0':
        _type = 'sne_luminosity_distance_lcdm_with_H0'
    elif type == 'sne_distmod_with_H0':
        _type = 'sne_lcdm_distmod_with_H0'
    elif type == 'combined_luminosity_distance_with_H0':
        _type = 'combined_luminosity_distance_lcdm_with_H0'
    elif type == 'combined_distmod':
        _type = 'combined_distmod_lcdm'
    elif type == 'combined_distmod_with_H0':
        _type = 'combined_distmod_lcdm_with_H0'
    else:
        _type = type
    return _type


def model_to_lcdm_cost_function(omm, z, data_obs, data_err, type='sne_lcdm_distmod', H0=70.0):
    _type = type_to_lcdm(type)

    return -log_likelihood(omm, z, data_obs, data_err, type=_type, H0=H0)


def model_to_lcdm_cost_function_with_H0(params, z, data_obs, data_err, type='sne_lcdm_distmod_with_H0'):
    _type = type_to_lcdm(type)
    return -log_likelihood(params, z, data_obs, data_err, type=_type)


def data_obs_from_waw0_cosmology(z, w0, wa, om0, type_for_cosmo, model_M=0 ,H0=70.0):
    _type = type_to_lcdm(type_for_cosmo)
    w0wacosmology = Flatw0waCDM(H0=H0, Om0=om0, w0=w0, wa=wa)
    if _type == 'sne_luminosity_distance_lcdm' or _type == 'sne_luminosity_distance_lcdm_with_H0':
        data_obs = sne_luminosity_distance(z=z, H0=H0, Om0=om0, w0=w0, wa=wa, cosmo=w0wacosmology)
    elif _type == 'sne_lcdm_distmod' or _type == 'sne_lcdm_distmod_with_H0':
        data_obs = sne_distmod(z=z, H0=H0, Om0=om0, w0=w0, wa=wa, cosmo=w0wacosmology , model_M = model_M)
    elif _type == 'bao_fap' or _type == 'fap_with_H0':
        data_obs = bao_fap(z=z, H0=H0, Om0=om0, w0=w0, wa=wa)
    elif _type == 'bao_dv_dr' or _type == 'dv_with_H0':
        data_obs = bao_dv_dr(z=z, H0=H0, Om0=om0, w0=w0, wa=wa)
    elif _type == 'combined_luminosity_distance_lcdm' or _type == 'combined_luminosity_distance_lcdm_with_H0':
        data_obs = {}
        if 'sne' in z:
            z_sne = z['sne']
            data_obs['sne'] = sne_luminosity_distance(z_sne, H0=H0, Om0=om0, w0=w0, wa=wa, cosmo=w0wacosmology)
        if 'dv' in z:
            z_dv = z['dv']
            data_obs['dv'] = bao_dv_dr(z_dv, H0=H0, Om0=om0, w0=w0, wa=wa)
        if 'fap' in z:
            z_fap = z['fap']
            data_obs['fap'] = bao_fap(z_fap, H0=H0, Om0=om0, w0=w0, wa=wa)
    elif _type == 'combined_distmod_lcdm' or _type == 'combined_distmod_lcdm_with_H0':
        data_obs = {}
        if 'sne' in z:
            z_sne = z['sne']
            data_obs['sne'] = sne_distmod(z_sne, H0=H0, Om0=om0, w0=w0, wa=wa, cosmo=w0wacosmology, model_M = model_M)
        if 'dv' in z:
            z_dv = z['dv']
            data_obs['dv'] = bao_dv_dr(z_dv, H0=H0, Om0=om0, w0=w0, wa=wa)
        if 'fap' in z:
            z_fap = z['fap']
            data_obs['fap'] = bao_fap(z_fap, H0=H0, Om0=om0, w0=wa, wa=wa)
    else:
        raise ValueError(f"Type {_type} not recognized")
    return data_obs


def worker_run_lcdm_om0(args):
    params, z, data_err, data_type = args
    omm = params[2]
    data_obs = data_obs_from_waw0_cosmology(z=z, w0=params[0], wa=params[1], om0=params[2], type_for_cosmo=data_type)
    result = minimize(model_to_lcdm_cost_function, omm, args=(z, data_obs, data_err, data_type), method='Nelder-Mead')
    return params[2], result.x


def run_lcdm_om0_by_multiprocessing(param_combinations, z, data_err, data_type):
    args = [(params, z, data_err, data_type) for params in param_combinations]
    results = []
    with get_context("spawn").Pool() as pool:
        for result in tqdm(pool.imap_unordered(worker_run_lcdm_om0, args), total=len(args), desc="Processing"):
            results.append(result)
    params = [res[0] for res in results]
    result_om0 = [res[1] for res in results]
    return params, result_om0


def worker_run_gussian_om0(args):
    params, z, data_err = args

    zbao = {'dv': z['dv'], 'fap': z['fap']}
    data_err_bao = {'dv': data_err['dv'], 'fap': data_err['fap']}

    snez = np.array(z['sne'])
    errsne = np.array(data_err['sne'])

    omm_initial = params[2]

    data_obs_sne = data_obs_from_waw0_cosmology(z=snez, w0=params[0], wa=params[1], om0=params[2],
                                                type_for_cosmo='sne_distmod')
    data_obs_bao = data_obs_from_waw0_cosmology(z=zbao, w0=params[0], wa=params[1], om0=params[2],
                                                type_for_cosmo='combined_distmod_lcdm')

    omm_min_sne = minimize(model_to_lcdm_cost_function, omm_initial, args=(snez, data_obs_sne, errsne, 'sne_distmod'),
                           method='Nelder-Mead').x[0]
    omm_min_bao = \
        minimize(model_to_lcdm_cost_function, omm_initial, args=(zbao, data_obs_bao, data_err_bao, 'combined_distmod'),
                 method='Nelder-Mead').x[0]

    epsilon = np.sqrt(np.finfo(float).eps)  # Machine epsilon for float64

    hessian_sne = np.zeros((1, 1))
    hessian_bao = np.zeros((1, 1))

    hessian_sne[0, 0] = (model_to_lcdm_cost_function(omm_min_sne + epsilon, snez, data_obs_sne, errsne, 'sne_distmod') -
                         2 * model_to_lcdm_cost_function(omm_min_sne, snez, data_obs_sne, errsne, 'sne_distmod') +
                         model_to_lcdm_cost_function(omm_min_sne - epsilon, snez, data_obs_sne, errsne,
                                                     'sne_distmod')) / (epsilon ** 2)

    hessian_bao[0, 0] = (model_to_lcdm_cost_function(omm_min_bao + epsilon, zbao, data_obs_bao, data_err_bao,
                                                     'combined_distmod') -
                         2 * model_to_lcdm_cost_function(omm_min_bao, zbao, data_obs_bao, data_err_bao,
                                                         'combined_distmod') +
                         model_to_lcdm_cost_function(omm_min_bao - epsilon, zbao, data_obs_bao, data_err_bao,
                                                     'combined_distmod')) / (epsilon ** 2)

    if np.isinf(hessian_sne).any() or np.isnan(hessian_sne).any():
        print(f"Invalid Hessian matrix for SNe at params: {params}")
        return np.nan, np.nan, np.nan, np.nan, params

    if np.isinf(hessian_bao).any() or np.isnan(hessian_bao).any():
        print(f"Invalid Hessian matrix for BAO at params: {params}")
        return np.nan, np.nan, np.nan, np.nan, params

    try:
        cov_matrix_sne = inv(hessian_sne)
        cov_matrix_bao = inv(hessian_bao)
    except LinAlgError:
        print(f"Singular matrix for params: {params}")
        return np.nan, np.nan, np.nan, np.nan, params

    omm_std_sne = np.sqrt(np.diag(cov_matrix_sne))[0]
    omm_std_bao = np.sqrt(np.diag(cov_matrix_bao))[0]

    return omm_min_sne, omm_min_bao, omm_std_sne, omm_std_bao, params


def worker_run_gussian_om0_with_H0(args):
    params, z, data_err, cov_matrix_sne = args

    zbao = {'dv': z['dv'], 'fap': z['fap']}
    data_err_bao = {'dv': data_err['dv'], 'fap': data_err['fap']}

    snez = np.array(z['sne'])
    errsne = np.array(data_err['sne'])

    h0_initial = params[3]
    omm_initial = params[2]

    data_obs_sne = data_obs_from_waw0_cosmology(z=snez, H0=params[3], om0=params[2], w0=params[0], wa=params[1],
                                                type_for_cosmo='sne_distmod_with_H0')
    data_obs_bao = data_obs_from_waw0_cosmology(z=zbao, H0=params[3], om0=params[2], w0=params[0], wa=params[1],
                                                type_for_cosmo='combined_distmod_with_H0')

    result_sne = minimize(model_to_lcdm_cost_function_with_H0, np.array([omm_initial, h0_initial]),
                          args=(snez, data_obs_sne, errsne, 'sne_distmod_with_H0', cov_matrix_sne, ),
                          method='BFGS')
    result_bao = minimize(model_to_lcdm_cost_function_with_H0, np.array([omm_initial, h0_initial]),
                          args=(zbao, data_obs_bao, data_err_bao,  'combined_distmod_with_H0',cov_matrix_sne,),
                          method='BFGS')

    omm_min_sne, h0_min_sne = result_sne.x
    omm_min_bao, h0_min_bao = result_bao.x

    if ((omm_min_sne > 2.95 or omm_min_sne < 0.05) or (
            h0_min_sne > 98 or h0_min_sne < 52)) or result_sne.success == False:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, params
    if ((omm_min_bao > 2.95 or omm_min_bao < 0.05) or (
            h0_min_bao > 98 or h0_min_bao < 52)) or result_bao.success == False:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, params

    epsilon = np.sqrt(np.finfo(float).eps)  # Machine epsilon for float64

    hessian_sne_test = np.zeros((2, 2))
    hessian_bao_test = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            params_eps = [omm_min_sne, h0_min_sne]
            params_eps[i] += epsilon
            params_eps[j] += epsilon
            f1 = model_to_lcdm_cost_function_with_H0(params_eps, snez, data_obs_sne, errsne, 'sne_distmod_with_H0')

            params_eps[j] -= 2 * epsilon
            f2 = model_to_lcdm_cost_function_with_H0(params_eps, snez, data_obs_sne, errsne, 'sne_distmod_with_H0')

            params_eps[i] -= 2 * epsilon
            f3 = model_to_lcdm_cost_function_with_H0(params_eps, snez, data_obs_sne, errsne, 'sne_distmod_with_H0')

            params_eps[j] += 2 * epsilon
            f4 = model_to_lcdm_cost_function_with_H0(params_eps, snez, data_obs_sne, errsne, 'sne_distmod_with_H0')

            hessian_sne_test[i, j] = (f1 - f2 - f3 + f4) / (4 * epsilon ** 2)

    for i in range(2):
        for j in range(2):
            params_eps = [omm_min_bao, h0_min_bao]
            params_eps[i] += epsilon
            params_eps[j] += epsilon
            f1 = model_to_lcdm_cost_function(params_eps, zbao, data_obs_bao, data_err_bao, 'combined_distmod_with_H0')

            params_eps[j] -= 2 * epsilon
            f2 = model_to_lcdm_cost_function(params_eps, zbao, data_obs_bao, data_err_bao, 'combined_distmod_with_H0')

            params_eps[i] -= 2 * epsilon
            f3 = model_to_lcdm_cost_function(params_eps, zbao, data_obs_bao, data_err_bao, 'combined_distmod_with_H0')

            params_eps[j] += 2 * epsilon
            f4 = model_to_lcdm_cost_function(params_eps, zbao, data_obs_bao, data_err_bao, 'combined_distmod_with_H0')

            hessian_bao_test[i, j] = (f1 - f2 - f3 + f4) / (4 * epsilon ** 2)

    if np.isinf(hessian_sne_test).any() or np.isnan(hessian_sne_test).any():
        print(f"Invalid Hessian matrix for SNe at params: {params}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, params

    if np.isinf(hessian_bao_test).any() or np.isnan(hessian_bao_test).any():
        print(f"Invalid Hessian matrix for BAO at params: {params}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, params

    try:
        inv(hessian_sne_test)
        inv(hessian_bao_test)
    except LinAlgError:
        print(f"Singular matrix for params: {params}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, params

    theta_map_sne = result_sne.x
    theta_map_bao = result_bao.x

    hessian_sne_at_map = hessian_neg_log_posterior(theta_map_sne, model_to_lcdm_cost_function_with_H0,
                                                   (snez, data_obs_sne, errsne, 'sne_distmod_with_H0', cov_matrix_sne))

    hessian_bao_at_map = hessian_neg_log_posterior(theta_map_bao, model_to_lcdm_cost_function_with_H0,
                                                   (zbao, data_obs_bao, data_err_bao,
                                                    'combined_distmod_lcdm_with_H0'))

    if np.isinf(hessian_sne_at_map).any() or np.isnan(hessian_sne_at_map).any():
        print(f"Invalid Hessian matrix for SNe at params: {params}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, params
    if np.isinf(hessian_bao_at_map).any() or np.isnan(hessian_bao_at_map).any():
        print(f"Invalid Hessian matrix for BAO at params: {params}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, params

    try:
        covariance_matrix_sne = inv(hessian_sne_at_map)
        covariance_matrix_bao = inv(hessian_bao_at_map)
    except LinAlgError:
        print(f"Singular matrix for params: {params}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, params

    sne_uncertainties = np.sqrt(np.diag(covariance_matrix_sne))
    omm_std_sne = sne_uncertainties[0]
    h0_std_sne = sne_uncertainties[1]

    bao_uncertainties = np.sqrt(np.diag(covariance_matrix_bao))
    omm_std_bao = bao_uncertainties[0]
    h0_std_bao = bao_uncertainties[1]

    return h0_min_sne, omm_min_sne, h0_std_sne, omm_std_sne, h0_min_bao, omm_min_bao, h0_std_bao, omm_std_bao, params


def hessian_neg_log_posterior(theta, cost_function, args, epsilon=1e-5):
    """
    Compute the Hessian matrix using numerical differentiation (finite differences).
    :param theta: Current parameters (best-fit)
    :param cost_function: Negative log-posterior (model function)
    :param args: Additional arguments for the cost function
    :param epsilon: Step size for finite differences
    :return: Hessian matrix
    """
    n_params = len(theta)
    hessian = np.zeros((n_params, n_params))

    grad = np.zeros(n_params)
    for i in range(n_params):
        theta_step = np.copy(theta)
        theta_step[i] += epsilon
        grad[i] = (cost_function(theta_step, *args) - cost_function(theta, *args)) / epsilon

    for i in range(n_params):
        for j in range(n_params):
            theta_ij = np.copy(theta)
            theta_ij[i] += epsilon
            theta_ij[j] += epsilon
            f_ij = cost_function(theta_ij, *args)

            theta_i = np.copy(theta)
            theta_i[i] += epsilon
            f_i = cost_function(theta_i, *args)

            theta_j = np.copy(theta)
            theta_j[j] += epsilon
            f_j = cost_function(theta_j, *args)

            f_base = cost_function(theta, *args)

            hessian[i, j] = (f_ij - f_i - f_j + f_base) / (epsilon ** 2)

    return hessian


def run_gaussian_methods_om0_by_multiprocessing(param_combinations, z, data_err):
    args = [(params, z, data_err) for params in param_combinations]
    results = []
    with get_context("spawn").Pool() as pool:
        for result in tqdm(pool.imap_unordered(worker_run_gussian_om0, args), total=len(args), desc="Processing"):
            results.append(result)
    omm_min_sne = [res[0] for res in results]
    omm_min_bao = [res[1] for res in results]
    omm_std_sne = [res[2] for res in results]
    omm_std_bao = [res[3] for res in results]
    param = [res[4] for res in results]
    return omm_min_sne, omm_min_bao, omm_std_sne, omm_std_bao, param


def run_gaussian_methods_om0_with_H0_by_multiprocessing(param_combinations, z, data_err,cov_matrix_sne=None):
    args = [(params, z, data_err, cov_matrix_sne) for params in param_combinations]
    results = []
    with get_context("spawn").Pool() as pool:
        for result in tqdm(pool.imap_unordered(worker_run_gussian_om0_with_H0, args), total=len(args),
                           desc="Processing"):
            results.append(result)

    h0_min_sne = [res[0] for res in results]
    omm_min_sne = [res[1] for res in results]
    h0_std_sne = [res[2] for res in results]
    omm_std_sne = [res[3] for res in results]
    h0_min_bao = [res[4] for res in results]
    omm_min_bao = [res[5] for res in results]
    h0_std_bao = [res[6] for res in results]
    omm_std_bao = [res[7] for res in results]
    param = [res[8] for res in results]

    return h0_min_sne, omm_min_sne, h0_std_sne, omm_std_sne, h0_min_bao, omm_min_bao, h0_std_bao, omm_std_bao, param


def calculate_om0_distributions(x, omm_min_sne, omm_min_bao, omm_std_sne, omm_std_bao, weights=None, n_jobs=-1):
    total_subtraction = np.zeros_like(x)

    combined_data = np.column_stack((omm_min_sne, omm_min_bao, omm_std_sne, omm_std_bao))

    out_of_range_mask = (combined_data[:, 0] < 0.003) | (combined_data[:, 0] > 0.997) | \
                        (combined_data[:, 1] < 0.003) | (combined_data[:, 1] > 0.997)

    combined_data[out_of_range_mask, :2] = np.nan
    omm_min_sne, omm_min_bao, omm_std_sne, omm_std_bao = combined_data.T

    def process_single(mu_sne, mu_bao, sigma_sne, sigma_bao, weight):
        if not np.isnan(mu_sne) and not np.isnan(mu_bao) and not np.isnan(sigma_sne) and not np.isnan(sigma_bao):
            return weight * norm.pdf(x, mu_sne - mu_bao, (sigma_sne ** 2 + sigma_bao ** 2) ** 0.5)
        return np.zeros_like(x)

    if weights is not None:
        weights = np.array(weights)
        if weights.size != len(omm_min_sne):
            raise ValueError("Weights should have the same size as the number of parameters")
        else:
            weights = weights / np.sum(weights)
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_single)(mu_sne, mu_bao, sigma_sne, sigma_bao, weight)
                for mu_sne, mu_bao, sigma_sne, sigma_bao, weight in
                tqdm(zip(omm_min_sne, omm_min_bao, omm_std_sne, omm_std_bao, weights), total=len(weights))
            )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single)(mu_sne, mu_bao, sigma_sne, sigma_bao, 1.0)
            for mu_sne, mu_bao, sigma_sne, sigma_bao in
            tqdm(zip(omm_min_sne, omm_min_bao, omm_std_sne, omm_std_bao), total=len(omm_min_sne))
        )

    for result in results:
        total_subtraction += result

    return total_subtraction


def worker_run_methods_om0(args):
    """

    :param args:
    :return:
    """
    params, z, data_err,sne_cov_matrix = args
    omm = params[2]
    zbao = {'dv': z['dv'], 'fap': z['fap']}
    data_err_bao = {'dv': data_err['dv'], 'fap': data_err['fap']}

    snez = np.array(z['sne'])
    errsne = np.array(data_err['sne'])

    data_obs_sne = data_obs_from_waw0_cosmology(z=snez, w0=params[0], wa=params[1], om0=params[2],
                                                type_for_cosmo='sne_distmod')
    data_obs_bao = data_obs_from_waw0_cosmology(z=zbao, w0=params[0], wa=params[1], om0=params[2],
                                                type_for_cosmo='combined_distmod')
    result_sne = minimize(model_to_lcdm_cost_function, omm, args=(snez, data_obs_sne, errsne, 'sne_distmod'),
                          method='Nelder-Mead')
    result_bao = minimize(model_to_lcdm_cost_function, omm, args=(zbao, data_obs_bao, data_err_bao, 'combined_distmod'),
                          method='Nelder-Mead')
    if not result_bao.success:
        print(f"Failed to find minimum for SNe data at {params}")
    if result_bao.x > 0.9:
        print(f"Large value for Om0 found {result_bao.x} at {params}, ")
    return params, result_sne.x, result_bao.x


def run_lcdm_methods_om0_by_multiprocessing(param_combinations, z, data_err,sne_cov_matrix=None):
    args = [(params, z, data_err,sne_cov_matrix) for params in param_combinations]
    results = []
    with get_context("spawn").Pool() as pool:
        for result in tqdm(pool.imap_unordered(worker_run_methods_om0, args), total=len(args), desc="Processing"):
            results.append(result)
    params = [res[0] for res in results]
    result_sne = [res[1] for res in results]
    result_bao = [res[2] for res in results]
    return params, result_sne, result_bao


def worker_run_methods_om0_with_H0(args):
    params, z, data_err, cov_matrix_sne = args

    zbao = {'dv': z['dv'], 'fap': z['fap']}
    data_err_bao = {'dv': data_err['dv'], 'fap': data_err['fap']}

    snez = np.array(z['sne'])
    errsne = np.array(data_err['sne'])

    h0_initial = params[3]
    omm_initial = params[2]

    data_obs_sne = data_obs_from_waw0_cosmology(z=snez, H0=params[3], om0=params[2], w0=params[0], wa=params[1],
                                                type_for_cosmo='sne_distmod_with_H0')
    data_obs_bao = data_obs_from_waw0_cosmology(z=zbao, H0=params[3], om0=params[2], w0=params[0], wa=params[1],
                                                type_for_cosmo='combined_distmod_with_H0')

    result_sne = minimize(model_to_lcdm_cost_function_with_H0, np.array([omm_initial, h0_initial]),
                          args=(snez, data_obs_sne, errsne, 'sne_distmod_with_H0',cov_matrix_sne),
                          method='BFGS')
    result_bao = minimize(model_to_lcdm_cost_function_with_H0, np.array([omm_initial, h0_initial]),
                          args=(zbao, data_obs_bao, data_err_bao, 'combined_distmod_with_H0'),
                          method='BFGS')

    omm_min_sne, h0_min_sne = result_sne.x
    omm_min_bao, h0_min_bao = result_bao.x

    if not result_sne.success:
        print(f"Failed to find minimum for SNe data at {params}")
    if not result_bao.success:
        print(f"Failed to find minimum for BAO data at {params}")

    return params, omm_min_sne, omm_min_bao


def run_lcdm_methods_om0_by_multiprocessing_with_H0(param_combinations, z, data_err,cov_matrix_sne=None):
    args = [(params, z, data_err,cov_matrix_sne) for params in param_combinations]
    results = []
    with get_context("spawn").Pool() as pool:
        for result in tqdm(pool.imap_unordered(worker_run_methods_om0_with_H0, args), total=len(args),
                           desc="Processing"):
            results.append(result)
    params = [res[0] for res in results]
    result_sne = [res[1] for res in results]
    result_bao = [res[2] for res in results]
    return params, result_sne, result_bao