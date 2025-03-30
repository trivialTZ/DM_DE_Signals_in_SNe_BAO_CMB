from procoli import lkl_prof

profile = lkl_prof(
  chains_dir='./', 
  # prof_param='w0_fld', 
  # prof_param='wa_fld', 
  # prof_param='Omega_m', 
  prof_param = 'H0',
  # info_root= 'mcmc_chains', 
)

profile.prof_max = 72. # 0, 1. , 0.33
profile.prof_min = 68. # -2, -3. , 0.27
profile.processes = 6

# run both increments! 
profile.prof_incr = -0.1 # 0.01, 0.02 , 0.002

# Run the global minimizer 

# Additional 1st step in minimizer because chains are not super converged 
profile.set_global_jump_fac([2, 1, 0.8, 0.5, 0.2, 0.1, 0.05])
profile.set_global_temp([0.5, 0.3333, 0.25, 0.2, 0.1, 0.005, 0.001])

profile.global_min(
  # run_glob_min=True, # if you don't pass this parameter, the code will 
                       # automatically decide whether to run the global minimizer. 
                       # It will run it if no global_min/global_min.bestfit 
                       # file exists or if the corresponding .log file has a worse
                       # -logLike than the info_root.log file 
  N_min_steps=4000 
) 


# Run the profile 

profile.init_lkl_prof()

profile.run_lkl_prof(
  time_mins=True, 
  N_min_steps=3000
)

# Run opposite increment 
profile.prof_incr = -0.1 # 0.01 w0, 0.02 wa, 0.0005 Omm, 0.05 H0

profile.global_min(
  N_min_steps=4000, 
  run_glob_min=False
) 

profile.init_lkl_prof()

profile.run_lkl_prof(
  time_mins=True, 
  N_min_steps=3000
)
