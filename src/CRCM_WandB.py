from backend.solver import MFE
from backend import CRCM, QRCM

import numpy as np
import wandb


# Set up WandB
wandb.login()

sweep_config = {
    'method'    : 'bayes',
    'name'      : 'CRCM sweep',
    'metric'    : {
        'name'  : 'MSE',
        'goal'  : 'minimize'
    },
    'parameters'    : {
        'epsilon'   : {'min': 0.0, 'max': 1.0},
        'tikhonov'  : {'min': 1e-10, 'max': 1.0},
        'N_units'   : {'min': 50, 'max': 10000},
        'connectivity'  : {'min': 0, 'max': 50}
    }
}

sweep_id = wandb.sweep(sweep_config, project="Quantum Turbulence Learning", entity="raihaan123")


# Data generation parameters
dt = 0.25

mfe_params = {
    "L_x": 4 * np.pi,
    "L_y": 2,
    "L_z": 2 * np.pi,
    "Re" : 400,
    "k_l": 0.48,
    "k_e": 0.1
}

# Define N for washout, training, validation and testing
N_transient     = 0
N_washout       = 100
N_train         = 200
N_test          = 200
N_sets          = (np.array([N_transient, N_washout, N_train, N_test]) / dt).astype(int)

u0 = [1, 0, 0.07066, -0.07076, 0, 0, 0, 0, 0]

# Instantiate the solver object with prescribed initial state
mfe9 = MFE(params=mfe_params, dt=dt, N_sets=N_sets, u0=u0)
mfe9.generate()

def train():
    with wandb.init() as run:
        config = wandb.config

        # Initialise the ESN
        crcm = CRCM(solver=mfe9,
                    N_units=config['N_units'],
                    connectivity=config['connectivity'],
                    eps=config['epsilon'],
                    tik=config['tikhonov'],
                    # Uncomment for sigmoid
                    activation=lambda x: 1/(1+np.exp(-x)),
                    seed=0)
        # Train the ESN with the training data
        crcm.train()
        crcm.forward()

        # Log data to WandB - can generate plots from this
        wandb.log({'Error Time Series': crcm.err_ts})
        wandb.log({'MSE': crcm.MSE})
        wandb.log({'Time': crcm.time})              # Log the time taken to train the model
        wandb.log({'Predicted': crcm.Y_pred})       # Log the predicted output


# Number of runs to execute
wandb.agent(sweep_id, function=train, count=100)