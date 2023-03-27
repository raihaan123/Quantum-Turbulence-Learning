from qiskit import QuantumCircuit, QuantumRegister, execute, Aer
import numpy as np
from numpy import pi, eye
from numpy.linalg import inv
import matplotlib.pyplot as plt

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
                        seed=0):

        # Checking the class has been initialized correctly - can be loosened later
        if solver is None:      raise ValueError("A solver object must be provided.")
        if eps < 0 or eps > 1:  raise ValueError("The leaking rate (eps) must be between 0 and 1.")
        if tik < 0:             raise ValueError("The Tikhonov regularization parameter (tik) must be non-negative.")


        """ Initialize the RCM """

        self.solver         = solver                        # Solver object - contains the dynamical system and the input data
        self.N_in           = solver.dim                    # Number of degrees of freedom of (possibly encoded) dynamical system

        self.seed           = seed                          # Set the seed for the random number generator
        self.rnd            = np.random.RandomState(seed)   # Random state object

        self.X              = np.zeros((solver.dim))        # Input state - X^t is a vector of dimension N_in
        self.psi            = None                          # psi^t is the latest reservoir state

        self.eps            = eps                           # Leaking rate epsilon -> P^(t+1) = epsilon*P_tilde^(t+1) + (1-epsilon)*P^(t)
        self.tik            = tik                           # Tikhonov regularization parameter
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

        u = self.solver.U[key]                              # Selecting the relevant input data

        with tqdm(total=u.shape[0], desc=key) as pbar:      # Add a status bar to the terminal
            for i, row in enumerate(u):
                pbar.update(1)
                self.X = row
                self.step()                                 # Propagate the state by one time step

                # Appending state to the reservoir state matrix during training
                if save:    self.R[:,i] = self.psi


    def closed_loop(self, ts, save=True):
        """ Execute the RCM in closed-loop mode - i.e with state feedback """

        self.Y_pred = np.zeros((ts, self.N_in))             # Reset the prediction register

        with tqdm(total=ts, desc="Closed Loop") as pbar:    # Add a status bar to the terminal
            for i in range(ts):
                pbar.update(1)
                self.step()                                 # Propagate the state by one time step
                self.X = np.dot(self.W_out, self.psi)       # Calculate the output state

                if save:    self.Y_pred[i,:] = self.X       # Log the output state


    def train(self):
        """ Train the RCM

        Saves:
            self.W_out (np.ndarray): Output weight matrix (optimized via ridge regression)
        """

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

        self.R = np.zeros((self.N_dof, len(U_train)))       # Reservoir state matrix [N_dof x N_train]

        ### Training pipeline ###
        self.open_loop("Washout")                           # Washout the reservoir
        self.open_loop("Train", save=True)                  # Evolve in open-loop mode for the training set with logging

        # Calculate the optimal output weight matrix using Ridge Regression. Dimensions [N_in x N_dof]
        self.W_out = np.dot(Y_train.T, np.dot(self.R.T, inv(np.dot(self.R, self.R.T) + self.tik * eye(self.N_dof))))


    def forward(self):
        # Run the test data through the reservoir, in closed loop
        self.Y_pred = np.zeros((len(self.solver.U["Test"]), self.N_in))
        self.closed_loop(len(self.solver.U["Test"]))

        # Find MSE
        Y_test = self.solver.Y["Test"]
        self.err_ts = np.abs(Y_test - self.Y_pred)/Y_test * 100
        self.MSE = np.mean(self.err_ts**2)

        print(f"MSE: {self.MSE}")

        # Plot the results
        self.plot_results(Y_test)

        # Save the figure
        plt.savefig(f"..\FYP Logbook\Diagrams\{self.__class__.__name__}_{self.solver.__class__.__name__}.png", dpi=500)
        plt.show()


    def plot_results(self, Y_test):
        plt.figure()
        plt.plot(self.Y_pred[:, 0], label="Predicted", color="red")
        plt.plot(self.Y_pred[:, 1:], color="red")

        plt.plot(Y_test[:, 0], label="Expected", color="blue")
        plt.plot(Y_test[:, 1:], color="blue")

        plt.title(f"{self.solver.__class__.__name__} system - {self.__class__.__name__}")
        plt.legend()
