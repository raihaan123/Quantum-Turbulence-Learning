from ode import generate_data, ddt_lorentz, ddt_MFE
from CRCM import CRCM
from QRCM import QRCM

import os;  os.environ["OMP_NUM_THREADS"] = '32' # Imposes cores

### Currently unused imports ###

# import h5py
# import skopt
# from skopt.space import Real
# from skopt.learning import GaussianProcessRegressor as GPR
# from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
# from scipy.io import loadmat, savemat
# from skopt.plots import plot_convergence


# Data generation parameters
dim             = 3
upsample        = 2                     # To increase the dt of the ESN wrt the numerical integrator
dt              = 0.005 * upsample      # Time step

# Solve the ODE system using generate_data() from ode.py
# data = generate_data(dim, upsample, dt, ddt=ddt_lorentz, noisy=True)

# # Initialise the ESN
# crcm = CRCM(dim=dim,
#             N_units=200,
#             connectivity=3,
#             seed=0)

# # Train the ESN with the training data
# crcm.train(data)

# # Test the ESN with the test data
# Y_pred = crcm.forward(U_test)


# Initialise the QRCM
qrcm = QRCM(dim=dim)

# Just testing here - this wil be self contained in the train and forward methods
qrcm.build_circuit()