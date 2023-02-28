from ode import generate_data, ddt_lorentz
from CRCM import CRCM

import os;  os.environ["OMP_NUM_THREADS"] = '32' # Imposes cores

### Currently unused imports ###

# import numpy as np
# import matplotlib.pyplot as plt
# import h5py
# import skopt
# from skopt.space import Real
# from skopt.learning import GaussianProcessRegressor as GPR
# from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
# from scipy.io import loadmat, savemat
# import time
# from skopt.plots import plot_convergence


# Data generation parameters
dim             = 3
upsample        = 2                     # To increase the dt of the ESN wrt the numerical integrator
dt              = 0.005 * upsample      # Time step

# Solve the ODE system using generate_data() function
U_tv, Y_tv, U_test, Y_test = generate_data(dim, upsample, dt, ddt=ddt_lorentz, noisy=True)

# Train the ESN
crcm = CRCM(N_units=200,
            connectivity=3,
            tikh = 1e-6),
            seed=0,
            ensemble=1)

crcm.train(U_tv, Y_tv)