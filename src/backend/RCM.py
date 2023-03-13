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
                        eps=1e-1,
                        tik=1e-6,
                        seed=0,
                        plot=False):

        """ Initialize the RCM """

        self.solver         = solver                        # Solver object - contains the dynamical system and the input data
        self.N_in           = solver.dim                    # Number of degrees of freedom of (possibly encoded) dynamical system

        self.seed           = seed                          # Set the seed for the random number generator
        self.rnd            = np.random.RandomState(seed)   # Random state object

        self.X              = np.zeros((solver.dim))        # Input state - X^t is a vector of dimension N_in
        self.psi            = None                          # psi^t is the latest reservoir state

        self.eps            = eps                           # Leaking rate epsilon -> P^(t+1) = epsilon*P_tilde^(t+1) + (1-epsilon)*P^(t)
        self.tik            = tik                           # Tikhonov regularization parameter
        self.plot           = plot                          # Plot the circuit if True
        self.time           = None                          # Time taken to complete the simulation - set by @hyperparameters decorator


    def refresh(self):
        """ Reset the reservoir state to initial state """
        self.rnd = np.random.RandomState(self.seed)
        self.psi = np.zeros(self.N_dof)

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


    def train(self, override=False):
        """ Train the RCM

        Saves:
            self.W_out (np.ndarray): Output weight matrix (optimized via ridge regression)
        """

        # Generate the training data
        self.solver.generate(override)

        # Load data
        U_washout       = self.solver.U["Washout"]
        U_train         = self.solver.U["Train"]
        Y_train         = self.solver.Y["Train"]
        U_test          = self.solver.U["Test"]
        Y_test          = self.solver.Y["Test"]

        self.u_mean     = self.solver.u_mean
        self.norm       = self.solver.norm

        self.bias_in    = self.solver.bias_in
        self.bias_out   = self.solver.bias_out

        # Washout the reservoir
        self.open_loop("Washout")

        # Target output matrix
        self.R = np.zeros((self.N_dof, len(U_train)))       # Dimensions [N_dof x N_train]

        # For each row in U_train, build and run the circuit - save the reservoir state |psi^(t+1)> in R
        self.open_loop("Train", save=True)

        # Calculate the optimal output weight matrix using Ridge Regression. Dimensions [N_in x N_dof]
        self.W_out = np.dot(Y_train.T, np.dot(self.R.T, np.linalg.inv(np.dot(self.R, self.R.T) + self.tik * np.eye(self.N_dof))))


    def forward(self):
        # Run washout first in open loop
        self.refresh()
        print(f"\nReservoir refreshed")
        self.open_loop("Washout")

        # Run the training data through the reservoir, again in open loop
        self.R = np.zeros((self.N_dof, len(self.solver.U["Train"])))       # Dimensions [N_dof x N_train]

        self.open_loop("Train", save=True)
        Y_train_pred = np.dot(self.W_out, self.R).T

        # Now run the test data through the reservoir, in closed loop
        self.Y_pred = np.zeros((len(self.solver.U["Test"]), self.N_in))
        self.closed_loop(len(self.solver.U["Test"]))

        # Find MSE
        Y_test = self.solver.Y["Test"]
        self.err_ts = np.abs(Y_test - self.Y_pred)/Y_test * 100
        self.MSE = np.mean(self.err_ts**2)

        print(f"MSE: {self.MSE}")

        # Plot the data in Y_train_pred
        plt.figure()
        plt.plot(self.Y_pred[:, 0], label="Predicted", color="red")
        plt.plot(self.Y_pred[:, 1:], color="red")

        # Overlay the expected output
        plt.plot(Y_test[:, 0], label="Expected", color="blue")
        plt.plot(Y_test[:, 1:], color="blue")

        plt.title(f"Lorenz system - {self.__class__.__name__}")
        plt.legend()

        # Save the figure - dpi of 300 is good for printing
        plt.savefig("..\FYP Logbook\Diagrams\CRCM_Lorenz.png", dpi=300)
        plt.show()