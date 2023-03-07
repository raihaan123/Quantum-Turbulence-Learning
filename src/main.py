from backend.solver import Lorentz, MFE
from backend import CRCM, QRCM

import numpy as np
import wandb
import os;  os.environ["OMP_NUM_THREADS"] = '32' # Imposes cores


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
        'qubits'    : {'values': [2, 4, 6]},
        'epsilon'   : {'min': 0.0, 'max': 1.0},
        'tikhonov'  : {'min': 1e-10, 'max': 1e-2},
        'N_train'   : {'values': [50, 100, 300, 500, 1000, 2000]},
        'N_test'    : {'values': [50, 100, 300, 500, 1000, 2000]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="Quantum Turbulence Learning", entity="raihaan123")


# Data generation parameters
dim             = 3
upsample        = 1                     # To increase the dt of the ESN wrt the numerical integrator
dt              = 0.005 * upsample      # Time step
params          = [8/3, 28, 10]         # Parameters for the Lorenz system

def train():
    with wandb.init() as run:
        config = wandb.config

        # Define N for washout, training, validation and testing
        N_washout       = 50
        N_train         = config['N_train']
        N_test          = config['N_test']
        N_sets          = [N_washout, N_train, N_test]

        # Instantiate the solver object
        lor3 = Lorentz(params, dt, N_sets)

        # Instantiate the QRCM object
        qrcm = QRCM(solver  = lor3,
                    qubits  = config['qubits'],
                    eps     = config['epsilon'],
                    tik     = config['tikhonov'])

        # Train the QRCM with the training data
        qrcm.train(override=True)

        # Log data to WandB - can generate plots from this
        wandb.log({'Error Time Series': qrcm.err_ts})
        wandb.log({'MSE': qrcm.MSE})
        wandb.log({'Time': qrcm.time})              # Log the time taken to train the model


# Number of runs to execute
wandb.agent(sweep_id, function=train, count=100)