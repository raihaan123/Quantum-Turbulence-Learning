from ode import generate_data, ddt_lorentz, ddt_MFE
from CRCM import CRCM
from QRCM import QRCM
import numpy as np
import wandb

import os;  os.environ["OMP_NUM_THREADS"] = '32' # Imposes cores


# # Set up WandB
# wandb.login()
# wandb.init(project="Quantum Turbulence Learning")

# sweep_config = {
#     'method': 'bayes',
#     'name': 'QRCM sweep',
#     'metric': {
#         'name': 'MSE',
#         'goal': 'minimize'
#         },
#     'parameters': {
#         'qubits': {'values': [2:8]},
#         'epsilon': {'min': 0, 'max': 1},
#         'tikhonov': {'min': 1e-10, 'max': 1e-2},
        
#     }
# }


# Data generation parameters
dim             = 3
upsample        = 1                     # To increase the dt of the ESN wrt the numerical integrator
dt              = 0.005 * upsample      # Time step

# Define N for washout, training, validation and testing
N_washout       = 50
N_train         = 100
N_test          = 200
N_sets          = [N_washout, N_train, N_test]

# Solve the ODE system using generate_data() from ode.py
data = generate_data(dim, N_sets, upsample, dt, ddt=ddt_lorentz, noisy=False, override=True)

# Initialise the QRCM
qrcm = QRCM(qubits=4,
            eps=0.05,
            tik=1e-6)

# Train the QRCM with the training data
qrcm.train(data)















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