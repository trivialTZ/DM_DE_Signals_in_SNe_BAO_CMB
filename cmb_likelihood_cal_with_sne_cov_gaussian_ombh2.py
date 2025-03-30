import numpy as np
from multiprocessing import get_context, Pool
import emcee
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
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
from wowadis_muilt_ombh2_gaussian_withcovmat import  (log_likelihood,worker_run_gussian_om0_with_H0,
                                                      hessian_neg_log_posterior,type_to_lcdm,
                                                      run_lcdm_methods_om0_by_multiprocessing,
                                                      worker_run_methods_om0_with_H0,sne_distmod,bao_dv_dr,bao_fap)
from cmb_v1_gaussian_ombh2 import CMBParams, set_astropy_cosmology, calculate_z_star,la_rzs_from_cosmoparams


def log_likelihood_cmb(params, data_obs_cmb, type):
    # data_obs_cmb =  l_A_obs, R_obs

    H0 = 70

    if type in ['combined_luminosity_distance_lcdm', 'sne_luminosity_distance_lcdm', 'sne_lcdm_distmod', 'combined_distmod_lcdm']:
        Om0 = params[0]
        M = params[1] if len(params) > 1 else None
        w0 = -1
        wa = 0
    elif type in ['combined_luminosity_distance_lcdm_with_H0', 'sne_luminosity_distance_lcdm_with_H0', 'sne_lcdm_distmod_with_H0', 'combined_distmod_lcdm_with_H0']:
        Om0 = params[0]
        H0 = params[1]
        M = params[2] if len(params) > 2 else None
        w0 = -1
        wa = 0
    elif type in ['sne_luminosity_distance_with_H0', 'sne_distmod_with_H0', 'dv_with_H0', 'fap_with_H0', 'combined_luminosity_distance_with_H0', 'combined_distmod_with_H0']:
        Om0 = params[0]
        w0 = params[1]
        wa = params[2]
        H0 = params[3]
        M = params[4] if len(params) > 4 else None
    else:
        Om0, w0, wa,M = params

    if not (0.03 <= Om0 <= 1.0):
        return -np.inf
    if not (-3 <= w0 <= 1):
        return -np.inf
    if not (-10 <= wa <= 5):
        return -np.inf
    if not (50 <= H0 <= 100):
        return -np.inf

    la_th, r_th = la_rzs_from_cosmoparams(H0=H0, omegam=Om0, omegabh2=None, omk=None, omnuh2=None, w=w0, wa=wa, nnu=None)

#    sigma_R = 0.0074
#    sigma_lA = 0.14

#    correlation_matrix = np.array([
#    [1.0, 0.54, -0.75, -0.79],
#    [0.54, 1.0, -0.42, -0.43],
#    [-0.75, -0.42, 1.0, 0.59],
#    [-0.79, -0.43, 0.59, 1.0]
#])    #R la ..... ....
#    correlation_matrix = correlation_matrix[:2, :2]

    cov_matrix = np.array([
        [0.000043, 0.000328],  # Var(R), Cov(R, l_A)
        [0.000328, 0.011046]  # Cov(R, l_A), Var(l_A)
    ])

    l_A_obs, R_obs = data_obs_cmb
#    sigma_vec = np.array([sigma_R, sigma_lA])
#    covariance_matrix = np.outer(sigma_vec, sigma_vec) * correlation_matrix

    cov_inv = np.linalg.inv(cov_matrix)
    delta = np.array([r_th - R_obs, la_th - l_A_obs])

    chi2 = delta.T @ cov_inv @ delta
    #ln_det_cov = np.linalg.slogdet(cov_matrix)[1]
    #ln_det_cov = -14.816830995273152
    ln_L = -0.5 * (chi2 + len(delta) * np.log(2 * np.pi) +14.816830995273152)
    #n_L = -0.5 * chi2

    #chi2_R = ((r_th - R_obs) / sigma_R) ** 2
    #chi2_lA = ((la_th - l_A_obs) / sigma_lA) ** 2

    #chi2 = chi2_R + chi2_lA
    return ln_L

def cmb_to_lcdm_cost_function(params, data_obs_cmb, type):
    _type = type_to_lcdm(type)
    return -log_likelihood_cmb(params, data_obs_cmb, type=_type)


def log_likelihood_with_cmb(params, z, data_obs, data_err, data_obs_cmb, type, cov_matrix_sne=None):
    l1 = log_likelihood(params, z, data_obs, data_err, type, cov_matrix_sne)
    l2 = log_likelihood_cmb(params, data_obs_cmb, type)
    return l1 + l2

def run_mcmc_with_cmb(z, data_obs, data_obs_cmb, data_err, type, initial_guess, ndim=3, nwalkers=50, nsteps=5000, nburn=100, nthin=15, cov_matrix_sne=None):
    pos = [initial_guess + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_with_cmb, args=(z, data_obs, data_err, data_obs_cmb, type, cov_matrix_sne), pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    samples = sampler.get_chain(discard=nburn, thin=nthin, flat=True)
    return samples

def log_likelihood_with_cmb_parallel(params_list, z, data_obs, data_err, data_obs_cmb, type, cov_matrix_sne=None):
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_log_likelihood)(param, z, data_obs, data_err, data_obs_cmb, type, cov_matrix_sne)
        for param in tqdm(params_list, desc="Evaluating log likelihood")
    )

    return results

def evaluate_log_likelihood(param, z, data_obs, data_err, data_obs_cmb, likelihood_type, cov_matrix_sne=None):

    # data_obs_cmb =  l_A_obs, R_obs
    log_likelihood_value = log_likelihood_with_cmb(param, z, data_obs, data_err, data_obs_cmb, likelihood_type, cov_matrix_sne)

    return {
        "w0": param[0],
        "wa": param[1],
        "omm": param[2],
        "H0": param[3],
        "log_likelihood": log_likelihood_value
    }


def run_mcmc_only_cmb(data_obs_cmb, initial_guess, type, ndim=4, nwalkers=50, nsteps=5000, nburn=100, nthin=15):
    pos = [initial_guess + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_cmb, args=(data_obs_cmb,type), pool=pool)

        sampler.run_mcmc(pos, nsteps, progress=True)

    samples = sampler.get_chain(discard=nburn, thin=nthin, flat=True)

    return samples


def worker_run_gussian_om0_with_H0_and_CMB(args):
    params, z, data_err, cov_matrix_sne = args

    w0 = params[0]
    wa = params[1]
    omm_initial = params[2]
    h0_initial = params[3]

    cmb_data_in_w0wa_obs_to_be_test_in_lcdm = la_rzs_from_cosmoparams(H0=h0_initial, omegam=omm_initial, omegabh2=None, omk=None, omnuh2=None, w=w0, wa=wa, nnu=None)
    if np.isnan(cmb_data_in_w0wa_obs_to_be_test_in_lcdm).any():
        return  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, params

    h0_min_sne, omm_min_sne, h0_std_sne, omm_std_sne, h0_min_bao, omm_min_bao, h0_std_bao, omm_std_bao, params = worker_run_gussian_om0_with_H0((params, z, data_err,cov_matrix_sne))

    result_cmb = minimize(cmb_to_lcdm_cost_function, np.array([omm_initial, h0_initial]),
                          args=(cmb_data_in_w0wa_obs_to_be_test_in_lcdm, 'combined_distmod_lcdm_with_H0'),
                          method='Nelder-Mead')
    omm_min_cmb, h0_min_cmb = result_cmb.x

    if ((omm_min_cmb > 2.95 or omm_min_cmb < 0.05) or (h0_min_cmb > 98 or h0_min_cmb < 52)) or result_cmb.success is False:
        print(f"Invalid CMB params: {params}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,params

    theta_map=result_cmb.x
    hessian_cmb_at_map = hessian_neg_log_posterior(theta_map, cmb_to_lcdm_cost_function,
                                                   (cmb_data_in_w0wa_obs_to_be_test_in_lcdm, 'combined_distmod_lcdm_with_H0'))

    if np.isinf(hessian_cmb_at_map).any() or np.isnan(hessian_cmb_at_map).any():
        print(f"Invalid Hessian matrix for CMB at params: {params}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, params

    try:
        covariance_matrix_cmb = inv(hessian_cmb_at_map)
    except LinAlgError:
        print(f"Singular matrix for CMB params: {params}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, params
    cmb_uncertainties = np.sqrt(np.diag(covariance_matrix_cmb))
    omm_std_cmb = cmb_uncertainties[0]
    h0_std_cmb = cmb_uncertainties[1]

    if np.isnan(h0_min_sne) or np.isnan(omm_min_sne) or np.isnan(h0_min_bao) or np.isnan(omm_min_bao):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, params

    return h0_min_sne, omm_min_sne, h0_std_sne, omm_std_sne, h0_min_bao, omm_min_bao, h0_std_bao, omm_std_bao, h0_min_cmb, omm_min_cmb, h0_std_cmb, omm_std_cmb, params

def run_gaussian_methods_om0_with_H0_and_CMB_by_multiprocessing(param_combinations, z, data_err,cov_matrix_sne=None):
    # Prepare the arguments by combining params, z, data_err, and data_obs_cmb for each parameter combination
    args = [(params, z, data_err,cov_matrix_sne) for params in param_combinations]

    results = []

    # Using multiprocessing to parallelize the computation
    with get_context("spawn").Pool() as pool:
        for result in tqdm(pool.imap_unordered(worker_run_gussian_om0_with_H0_and_CMB, args), total=len(args), desc="Processing"):
            results.append(result)

    # Collect the results from SNe, BAO, and CMB
    h0_min_sne = [res[0] for res in results]
    omm_min_sne = [res[1] for res in results]
    h0_std_sne = [res[2] for res in results]
    omm_std_sne = [res[3] for res in results]
    h0_min_bao = [res[4] for res in results]
    omm_min_bao = [res[5] for res in results]
    h0_std_bao = [res[6] for res in results]
    omm_std_bao = [res[7] for res in results]
    h0_min_cmb = [res[8] for res in results]
    omm_min_cmb = [res[9] for res in results]
    h0_std_cmb = [res[10] for res in results]
    omm_std_cmb = [res[11] for res in results]
    param = [res[12] for res in results]

    return h0_min_sne, omm_min_sne, h0_std_sne, omm_std_sne, h0_min_bao, omm_min_bao, h0_std_bao, omm_std_bao, h0_min_cmb, omm_min_cmb, h0_std_cmb, omm_std_cmb, param


def worker_run_methods_om0_with_cmb(args):
    """
    Worker function to run SNe, BAO, and CMB analysis for LCDM.
    Calls worker_run_methods_om0 for SNe and BAO, then adds CMB.
    """
    params, omm_min_sne, omm_min_bao = worker_run_methods_om0_with_H0(args)[:3]

    w0, wa, omm_initial, h0_initial= params[0], params[1], params[2], params[3]

    cmb_data_in_w0wa_obs_to_be_test_in_lcdm = la_rzs_from_cosmoparams(H0=h0_initial, omegam=omm_initial,
                                                                      omegabh2=None, omk=None, omnuh2=None, w=w0, wa=wa,
                                                                      nnu=None)
    result_cmb = minimize(cmb_to_lcdm_cost_function, np.array([omm_initial, h0_initial]),
                          args=(cmb_data_in_w0wa_obs_to_be_test_in_lcdm, 'combined_distmod_lcdm_with_H0'),
                          method='Nelder-Mead')
    omm_min_cmb, h0_min_cmb = result_cmb.x

    if not result_cmb.success:
        print(f"Failed to find minimum for CMB at {params}")

    return params, omm_min_sne, omm_min_bao, omm_min_cmb


def run_lcdm_methods_om0_with_cmb_by_multiprocessing(param_combinations, z, data_err, sne_cov_matrix=None):
    args = [(params, z, data_err,sne_cov_matrix) for params in param_combinations]
    results = []

    # Use multiprocessing to run SNe, BAO, and CMB analysis
    with get_context("spawn").Pool() as pool:
        for result in tqdm(pool.imap_unordered(worker_run_methods_om0_with_cmb, args), total=len(args), desc="Processing"):
            results.append(result)

    # Extract results for SNe, BAO, and CMB
    params = [res[0] for res in results]
    result_sne = [res[1] for res in results]
    result_bao = [res[2] for res in results]
    result_cmb = [res[3] for res in results]

    return params, result_sne, result_bao, result_cmb


def maybe_later(z, data_obs_cosmology_dict, data_obs):
    if 'sne' in z:
        z_sne = z['sne']
        data_obs_sne_cosmo = data_obs_cosmology_dict['sne']
        omm = data_obs_sne_cosmo['omm']
        H0 = 70
        w0 = None
        wa = None
        if 'H0' in data_obs_sne_cosmo:
            H0 = data_obs_sne_cosmo['H0']
        if 'w0' in data_obs_sne_cosmo:
            w0 = data_obs_sne_cosmo['w0']
        if 'wa' in data_obs_sne_cosmo:
            wa = data_obs_sne_cosmo['wa']
        data_obs_sne= sne_distmod(z=z_sne, H0=H0, Om0=omm, w0=w0, wa=wa)
        data_obs['sne'] = data_obs_sne
    if 'dv' in z:
        z_dv = z['dv']
        data_obs_dv_cosmo = data_obs_cosmology_dict['dv']
        omm = data_obs_dv_cosmo['omm']
        H0 = 70
        w0 = None
        wa = None
        if 'H0' in data_obs_dv_cosmo:
            H0 = data_obs_dv_cosmo['H0']
        if 'w0' in data_obs_dv_cosmo:
            w0 = data_obs_dv_cosmo['w0']
        if 'wa' in data_obs_dv_cosmo:
            wa = data_obs_dv_cosmo['wa']
        data_obs_dv = bao_dv_dr(z=z_dv, H0=H0, Om0=omm, w0=w0, wa=wa)
        data_obs['dv'] = data_obs_dv
    if 'fap' in z:
        z_fap = z['fap']
        data_obs_fap_cosmo = data_obs_cosmology_dict['fap']
        omm = data_obs_fap_cosmo['omm']
        H0 = 70
        w0 = None
        wa = None
        if 'H0' in data_obs_fap_cosmo:
            H0 = data_obs_fap_cosmo['H0']
        if 'w0' in data_obs_fap_cosmo:
            w0 = data_obs_fap_cosmo['w0']
        if 'wa' in data_obs_fap_cosmo:
            wa = data_obs_fap_cosmo['wa']
        data_obs_fap= bao_fap(z=z_fap, H0=H0, Om0=omm, w0=w0, wa=wa)
        data_obs['fap'] = data_obs_fap
    data_obs_cmb=la_rzs_from_cosmoparams(H0=70, omegam=0.3, w=-1.0, wa=0.0)
    return  None


