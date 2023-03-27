# %% [markdown]
# ## Quantum Reservoir Computing for Chaotic Dynamics
# FYP Masters project by Raihaan Usman

# %%
from backend.solver import Lorenz, MFE
from backend import CRCM, QRCM

import numpy as np
from matplotlib import pyplot as plt

# %% [markdown]
# #### Configuring the dynamical system

# %%
# Data generation parameters
upsample        = 1                     # To increase the dt of the ESN wrt the numerical integrator
dt              = 0.005 * upsample      # Time step
params          = [8/3, 28, 10]         # Parameters for the Lorenz system

# Define N for washout, training, validation and testing
N_washout       = 50
N_train         = 1000
N_test          = 100
N_sets          = [N_washout, N_train, N_test]

# Instantiate the solver object
lor3 = Lorenz(params, dt, N_sets)
lor3.generate(override=True)            # Can always be regenerated with an Autoencoder

# %% [markdown]
# #### CRCM for Lorenz system

# %%
# Initialise the ESN
# crcm = CRCM(solver=lor3,
#             N_units=500,
#             connectivity=10,
#             eps= 5e-2,
#             tik= 1e-4,
#             # sigmoid
#             activation=lambda x: 1/(1+np.exp(-x)),
#             seed=0)

# # Train the ESN with the training data
# crcm.train()
# crcm.forward()

# %% [markdown]
# #### QRCM for Lorenz system

# %%
# Instantiate the QRCM object
qrcm = QRCM(solver  = lor3,
            qubits  = 3,
            eps     = 5e-2,
            tik     = 1e-4,
            plot    = True)

# Train the QRCM with the training data
qrcm.train()
qrcm.forward()

# %%
# plot the first and second dimension of R vs time
plt.plot(qrcm.R[3,:])
print(qrcm.R.shape)



# %%
