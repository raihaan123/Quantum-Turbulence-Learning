from ode import generate_data, ddt_lorentz, ddt_MFE
from CRCM import CRCM
from QRCM import QRCM
import numpy as np

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
dt              = 0.01 * upsample      # Time step

# Solve the ODE system using generate_data() from ode.py
data = generate_data(dim, upsample, dt, ddt=ddt_lorentz, noisy=True)

# # Initialise the ESN
# crcm = CRCM(dim=dim,
#             N_units=200,
#             connectivity=3,
#             seed=0)

# # Train the ESN with the training data
# crcm.train(data)

# # Test the ESN with the test data
# U_washout   = data['U_washout']
# U_test      = data['U_test']
# Y_test      = data['Y_test']

# xf = crcm.open_loop(U_washout, np.zeros(200))[-1,: 200]
# Y_pred = crcm.open_loop(U_test, xf)

# # Plot the error time series for each component
# eror_ts = np.abs(Y_test - Y_pred)

# for i in range(dim):
#     plt.plot(err_ts[:,i], label=f"Dimension {i+1}")
# plt.title("Absolute Error Time Series")
# plt.legend()
# plt.xlabel("Time Step")
# plt.ylabel("Absolute Error")
# plt.show()


# Initialise the QRCM
qrcm = QRCM(dim=dim, plot=False)

# Train the QRCM with the training data
qrcm.train(data)
# qrcm.test(data)
