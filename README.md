# DM_DE_Signals_in_SNe_BAO_CMB

This repository contains the code and analysis scripts accompanying the paper:

**"Uniting the Observed Dynamical Dark Energy Preference with the Discrepancies in Ωₘ and H₀ Across Cosmological Probes"**  
[arXiv:2412.04430](https://arxiv.org/abs/2412.04430)

The project investigates whether an underlying dynamical dark energy (w₀wₐCDM) cosmology can naturally produce the observed discrepancies in the matter density parameter (Ωₘ) and the Hubble constant (H₀) when individual cosmological probes—Type Ia Supernovae (SNe), Baryon Acoustic Oscillations (BAO), and the Cosmic Microwave Background (CMB)—are analyzed under the ΛCDM framework.

---

## Repository Structure

- **notebooks**  
  Jupyter notebooks for:
  - Reproducing all key figures from the paper, including contour and residual plots
  - Step-by-step demonstration of likelihood construction and analysis pipeline

- **data**  
  Directory containing auxiliary data files used in simulations and plotting.

- **scripts** (main directory)  
  Python scripts for:
  - Constructing likelihoods (SNe, BAO, CMB)
  - Running simulations in ΛCDM and w₀wₐCDM cosmologies
  - Performing Fisher matrix and MCMC analysis
  
- (also Appendix F PROFILE LIKELIHOOD TESTS in 'tz_w0wa_profile_minimal_figure10' need [procoli](https://github.com/tkarwal/procoli))

---

#### External Files for Inv or SNe matrix & Appendix B (Figure 5)

For generating **Figure 5** in **Appendix B** of the paper (grid-based exploration of the w₀–wₐ parameter space), please also refer to the following three data files hosted at:

🔗 [Google Drive Folder (SNe matrix Figure 5 files)](https://drive.google.com/drive/folders/1AYAAuGNDkOIizO1JJkExIObTxeTr0zRL?usp=sharing)

- `fw0wacdm_SN+eBOSS+3x2pt.txt`
- `fw0wacdm_planck+SN+eBOSS+3x2pt.txt`
- `fw0wacdm_SN_emcee.txt`
- `inv_cov_matrix_sne.npy`

These files provide precomputed likelihood evaluations and MCMC samples used to generate the blue and orange contours in Figure 5.

---

## Dependencies

To run the code, install the following packages:

```
numpy
matplotlib
scipy
joblib
tqdm
pickle
corner
pandas
astropy
chainconsumer
emcee
```

You can install them with:

```bash
pip install numpy matplotlib scipy joblib tqdm pickle corner pandas astropy chainconsumer emcee
```

---

## Citation

If you use this code in your research, please cite:

> Tang, TZ, Brout, D., Karwal, T., Chang, C., Miranda, V., & Vincenzi, M. (2025). *Uniting the Observed Dynamical Dark Energy Preference with the Discrepancies in Ωₘ and H₀ Across Cosmological Probes*. [arXiv:2412.04430](https://arxiv.org/abs/2412.04430).
