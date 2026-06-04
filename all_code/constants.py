import numpy as np

OMEGA        = 2 * np.pi * 0.8   # mean Rabi frequency
OMEGA_SPREAD = 0.2               # relative spread of Omega across trajectories
KAPPA        = 0.4               # total measurement strength
ETA          = 0.3               # measurement efficiency (monitored fraction of KAPPA)
GAMMA_DECAY  = 0.1               # energy relaxation rate
dt    = 0.01
T     = 4.0
times = np.arange(0, T, dt)
Nt    = len(times)
