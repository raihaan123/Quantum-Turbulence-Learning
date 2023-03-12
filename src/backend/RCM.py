from qiskit import QuantumCircuit, QuantumRegister, execute, Aer
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from tqdm import tqdm

# Local imports
from tools.decorators import debug, hyperparameters


class RCM:
    """ Base class for a general RCM - Reservoir Computing Model

    Methods:
        - __init__(): Initialize the RCM.
        - open_loop(): Run the RCM in open-loop mode.
        - closed_loop(): Run the RCM in closed-loop mode.
        - train(): Train the RCM.
    """

    def __init__(self,  solver=None,
                        eps=1e-2,
                        tik=1e-2,
                        seed=0,
                        autoencoder=None,                   # N_latent <= N_in
                        plot=False):

        """ Initialize the RCM """

        self.solver         = solver                        # Solver object - contains the dynamical system and the input data
        self.N_in           = solver.dim                    # Number of degrees of freedom of (possibly encoded) dynamical system

        self.seed           = seed                          # Set the seed for the random number generator
        self.rnd            = np.random.RandomState(seed)   # Random state object

        self.X              = np.zeros((solver.dim))        # Input state - X^t is a vector of dimension N_in
        self.psi            = None                          # psi^t is the latest reservoir state

        self.eps            = eps                           # Leaking rate epsilon -> P^(t+1) = epsilon*P_tilde^(t+1) + (1-epsilon)*P^(t)
        self.tikhonov       = tik                           # Tikhonov regularization parameter
        self.plot           = plot                          # Plot the circuit if True
        self.time           = None                          # Time taken to complete the simulation - set by @hyperparameters decorator


    def refresh(self):
        """ Reset the reservoir state to initial state """
        self.rnd = np.random.RandomState(self.seed)


    def step(self):
        """ Propagate the RCM state by one time step """
        raise NotImplementedError


    def open_loop(self, key, save=False):
        """ Execute the RCM in open-loop mode -i.e no state feedback """

        # Add a status bar to the termainal - tqdm is a progress bar library
        with tqdm(total=self.solver.U[key].shape[0], desc=key) as pbar:
            for i, row in enumerate(self.solver.U[key]):
                pbar.update(1)
                self.X = row
                self.step()

                # Appending state to the reservoir state matrix during training
                if save:    self.R[:,i] = self.psi


    def closed_loop(self, ts, save=True):
        """ Execute the RCM in closed-loop mode - i.e with state feedback """

        # Add a status bar to the termainal - tqdm is a progress bar library
        with tqdm(total=ts, desc="Closed Loop") as pbar:
            for i in range(ts):
                pbar.update(1)
                self.step()
                self.X = np.dot(self.W_out, self.psi)

                # Calculate the output state
                if save:    self.Y_pred[i,:] = self.X


    # @hyperparameters
    def train(self, override=False):
        """ Train the RCM

        Saves:
            self.W_out (np.ndarray): Output weight matrix (optimized via ridge regression)
        """

        # Generate the training data
        self.solver.generate(override)

        # Load data
        U_washout = self.solver.U["Washout"]
        U_train   = self.solver.U["Train"]
        Y_train   = self.solver.Y["Train"]
        U_test    = self.solver.U["Test"]
        Y_test    = self.solver.Y["Test"]

        self.u_mean = self.solver.u_mean
        self.norm   = self.solver.norm

        # Washout the reservoir
        self.open_loop("Washout")

        # Display the current input row self.X and the current reservoir state self.psi
        print(f"\nX: {self.X} - supposed to be {U_washout[-1]}")
        print(f"psi: {self.psi}\n")

        # Target output matrix
        self.R = np.zeros((self.N_dof, len(U_train)))       # Dimensions [N_dof x N_train]
        print(f"\nPreparing to start training - shape of R: {self.R.shape}")

        # For each row in U_train, build and run the circuit - save the reservoir state |psi^(t+1)> in R
        self.open_loop("Train", save=True)

        # Display the current input row self.X and the current reservoir state self.psi
        print(f"\nX: {self.X} - supposed to be {U_train[-1]}")
        print(f"Shape of R: {self.R.shape}")
        print(f"R: {self.R}\n")

        # Calculate the optimal output weight matrix using Ridge Regression. Dimensions [N_in x N_dof]
        self.W_out = np.dot(Y_train.T, np.dot(self.R.T, np.linalg.inv(np.dot(self.R, self.R.T) + self.tikhonov * np.eye(self.N_dof))))

        # Print the shape and full matrix W_out
        print(f"\nShape of W_out: {self.W_out.shape}")
        print(f"W_out: {self.W_out}")


    def forward(self):
        # Run washout first in open loop
        self.refresh()
        print(f"\nReservoir refreshed")
        self.open_loop("Washout")

        # Display the current input row self.X and the current reservoir state self.psi
        print(f"psi: {self.psi}\n")

        # Run the training data through the reservoir, again in open loop
        self.R = np.zeros((self.N_dof, len(self.solver.U["Train"])))       # Dimensions [N_dof x N_train]
        print(f"\nPreparing to log training run - shape of R: {self.R.shape}")
        self.open_loop("Train", save=True)

        # Display the current input row self.X and the current reservoir state self.psi
        print(f"Shape of R: {self.R.shape}")
        print(f"R: {self.R}\n")

        R1 = self.R.copy()
        Y_train_pred = np.dot(self.W_out, R1).T
        print(f"\nY_train_pred: {Y_train_pred}")

        # # Now run the test data through the reservoir, in closed loop
        # self.Y_pred = np.zeros((len(self.solver.U["Test"]), self.N_in))
        # self.closed_loop(len(self.solver.U["Test"]))

        # # Find MSE
        # self.err_ts = np.abs(self.solver.Y["Test"] - self.Y_pred)/self.solver.Y["Test"] * 100
        # self.MSE = np.mean(self.err_ts**2)

        # print(f"MSE: {self.MSE}")