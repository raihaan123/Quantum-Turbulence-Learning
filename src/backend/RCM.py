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
                        autoencoder=None,       # N_latent <= N_in
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

                # Calculate the output state
                self.X = np.dot(self.W_out, self.psi)
                if save:    self.Y_pred[i,:] = self.X


    @hyperparameters
    def train(self, override=False):
        """ Train the QRCM

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

        # Washout the reservoir
        self.open_loop("Washout")

        # Target output matrix
        self.R = np.zeros((self.N_dof, len(U_train)))       # Dimensions [N_dof x N_train]

        # For each row in U_train, build and run the circuit - save the reservoir state |psi^(t+1)> in R
        self.open_loop("Train", save=True)

        # Calculate the optimal output weight matrix using Ridge Regression. Dimensions [N_in x N_dof]
        self.W_out = np.dot(Y_train.T, np.dot(self.R.T, np.linalg.inv(np.dot(self.R, self.R.T) + self.tikhonov * np.eye(self.N_dof))))

        self.R = np.zeros((self.N_dof, len(U_test)))        # Reset reservoir state matrix
        self.open_loop("Test", save=True)                   # Run the test data through the reservoir
        Y_pred = np.dot(self.W_out, self.R).T               # Multiply R by W_out to get all the output states

        self.err_ts = np.abs(Y_test - Y_pred)               # Calculate the absolute error for each timestep

        # Find MSE
        self.MSE = np.mean(self.err_ts**2)
        self.MSE_full = np.mean(self.err_ts**2, axis=0)


    def forward(self):
        # Run washout first in open loop
        self.open_loop("Washout")

        # Run the training data through the reservoir, again in open loop
        self.open_loop("Train")

        # Now run the test data through the reservoir, in closed loop
        self.Y_pred = np.zeros((len(self.solver.U["Test"]), self.N_in))
        self.closed_loop(len(self.solver.U["Test"]))

        # Plot the results over the test data
        plt.figure()
        for i in range(self.N_in):
            plt.plot(self.Y_pred[:,i], label=f"Predicted Dimension {i+1}")
            plt.plot(self.solver.Y["Test"][:,i], label=f"Target Dimension {i+1}")
        plt.title("Predicted vs Target Time Series")
        plt.legend()
        plt.xlabel("Time Step")
        plt.ylabel("State")
        plt.show()


        # Calculate the absolute error for each timestep
        # self.err_ts = np.abs(self.solver.Y["Test"] - self.Y_pred)

        # Plot the error time series for all N_in dimensions
        # plt.figure()
        # for i in range(self.N_in):
        #     plt.plot(self.err_ts[:,i], label=f"Dimension {i+1}")
        # plt.title("Absolute Error Time Series")
        # plt.legend()
        # plt.xlabel("Time Step")
        # plt.ylabel("Absolute Error")
        # plt.ylim(0, 1.1 * np.max(self.err_ts))        # make sure the data covers 70% of the plot height
        # plt.show()