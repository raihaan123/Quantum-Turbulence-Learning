from backend.solver import MFE, Lorenz
from backend import CRCM, QRCM

import numpy as np

# dt = 0.1

# mfe_params = {
#     "L_x": 4 * np.pi,
#     "L_y": 2,
#     "L_z": 2 * np.pi,
#     "Re" : 400,
#     "k_l": 0.48,
#     "k_e": 0.1
# }

# # Define N for washout, training, validation and testing
# N_transient     = 10
# N_washout       = 100
# N_train         = 300
# N_test          = 300
# N_sets          = (np.array([N_transient, N_washout, N_train, N_test]) / dt).astype(int)

# u0 = [1, 0, 0.07066, -0.07076, 0, 0, 0, 0, 0]

# # Instantiate the solver object with prescribed initial state
# mfe9 = MFE(params=mfe_params, dt=dt, N_sets=N_sets, u0=u0)
# mfe9.generate()

# # Instantiate the QRCM object
# qrcm = QRCM(solver  = mfe9,
#             qubits  = 3,
#             eps     = 5e-5,
#             tik     = 1e-5,
#             plot    = True)

# # Train the QRCM with the training data
# qrcm.train()
# qrcm.forward()




# Data generation parameters
upsample        = 1                     # To increase the dt of the ESN wrt the numerical integrator
dt              = 1e-2 * upsample       # Time step

lor_params      = {'beta' : 8/3,        # Parameters for the Lorenz system
                   'rho'  : 28,
                   'sigma': 10}

# Define N for transient, washout, training, validation and testing
N_transient     = 100
N_washout       = 10
N_train         = 50
N_test          = 5
N_sets          = (np.array([N_transient, N_washout, N_train, N_test]) / dt).astype(int)

# Instantiate the solver object with random initial state
lor3 = Lorenz(lor_params, dt/upsample, N_sets, noise=0)
lor3.generate()            # Can always be regenerated with an Autoencoder later

# Instantiate the QRCM object
qrcm = QRCM(solver  = lor3,
            qubits  = 3,
            eps     = 5e-2,
            tik     = 1e-4,
            # x0      = 7.5,
            # x1      = 9.5,
            # x2      = 5.3,
            plot    = True)

# Train the QRCM with the training data
qrcm.train()
qrcm.forward()