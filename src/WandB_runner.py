from backend.solver import MFE, Lorenz
from backend import CRCM, QRCM

import numpy as np
import wandb


# Set up WandB
wandb.login()

sweep_config = {
    'method'    : 'bayes',
    'name'      : 'QRCM sweep',
    'metric'    : {
        'name'  : 'MSE',
        'goal'  : 'minimize'
    },
    'parameters'    : {
        # 'epsilon'   : {'min': 0.0, 'max': 1.0},
        # 'tikhonov'  : {'min': 1e-10, 'max': 1e-2},
        'x0'        : {'min': 0.0, 'max': 10.0},
        'x1'        : {'min': 0.0, 'max': 10.0},
        'x2'        : {'min': 0.0, 'max': 10.0}
    }
}

sweep_id = wandb.sweep(sweep_config, project="Quantum Turbulence Learning", entity="raihaan123")


# Data generation parameters
upsample        = 10                    # To increase the dt of the ESN wrt the numerical integrator
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

def train():
    with wandb.init() as run:
        config = wandb.config

        # Instantiate the QRCM object
        qrcm = QRCM(solver  = lor3,
                    qubits  = 3,
                    eps     = 5e-2,
                    tik     = 1e-4,
                    x0      = config['x0'],
                    x1      = config['x1'],
                    x2      = config['x2'])

        # Train the QRCM with the training data
        qrcm.train()
        qrcm.forward()

        # Log data to WandB - can generate plots from this
        wandb.log({'Error Time Series': qrcm.err_ts})
        wandb.log({'MSE': qrcm.MSE})
        wandb.log({'Time': qrcm.time})              # Log the time taken to train the model


# Number of runs to execute
wandb.agent(sweep_id, function=train, count=100)