import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigs
import time

# Local imports
from tools.decorators import debug, hyperparameters

### Currently unused imports ###
# import h5py
# import skopt
# from skopt.space import Real
# from skopt.learning import GaussianProcessRegressor as GPR
# from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
# from scipy.io import loadmat, savemat
# from skopt.plots import plot_convergence


class CRCM:
    """ Class for the CRCM - Classical Reservoir Computing Model
    Heavily adapted from Alberto Racca's implementation: https://www.sciencedirect.com/science/article/pii/S0893608021001969

    Attributes:
        N_units: number of reservoir units
        N_in: number of input units
        N_out: number of output units

        Win: input weights
        W: reservoir weights
        Wout: output weights

    Methods:
        init: initializes the ESN with random Win and W
        step: advances one ESN time step
        open_loop: advances ESN in open-loop
        train: trains the ESN - ie optimizes Wout

    """

    def __init__(self, solver=None,
                       N_units=200,
                       N_splits=4,
                       tikh=1e-6,
                       sigma_in=1,
                       connectivity=3,
                       seed=0):

        self.solver         = solver
        self.N_units        = N_units
        self.N_splits       = N_splits
        self.tikh           = tikh
        self.sigma_in       = sigma_in
        self.connectivity   = connectivity
        self.sparseness     = 1 - connectivity/(N_units-1)

        self.seed           = seed
        rnd = self.rnd      = np.random.RandomState(seed)

        # Seting up the ESN with random Win
        self.Win = lil_matrix((N_units, dim+1))      # Sparse syntax for the input matrix, with +1 for bias input

        # Apply a random number in [-1, 1) to a randomn column in each row in Win
        for i in range(N_units):
            self.Win[i, rnd.randint(0, dim+1)] = rnd.uniform(-1, 1)

        # Convert to CSR format for faster matrix-vector multiplication
        Win = Win.tocsr()

        # On average only connectivity elements different from zero
        self.W = csr_matrix(rnd.uniform(-1, 1, (N_units, N_units)) * (rnd.rand(N_units, N_units) < (1-self.sparseness)))

        # The spectral radius of W is the maximum absolute value of its eigenvalues
        self.rho = np.abs(eigs(self.W, k=1, which='LM', return_eigenvectors=False))[0]

        # Rescale W to have a spectral radius of 1
        self.W *= 1/self.rho


    def step(self):
        """ Advances one ESN time step

            Args (self):
                psi: reservoir state
                u: input

            Saves:
                psi: new reservoir state
        """

        # Load class attributes
        u           = self.u
        sigma_in    = self.sigma_in
        rho         = self.rho

        # Input is normalized and input bias added
        u_augmented = np.hstack(((u-self.u_mean)/self.norm, self.bias_in))

        # Reservoir update - accessing the current reservoir state from the class attribute
        psi         = np.tanh(self.Win.dot(u_augmented*sigma_in) + self.W.dot(rho*self.psi))

        # Output bias added and state saved
        self.psi    = np.concatenate((psi, self.bias_out))


    def open_loop(self, ts):
        """ Advances ESN in open-loop.

            Returns:
                Xa: Time series of augmented reservoir states
        """

        # Output bias
        self.bias_out   = np.array([1.])

        x   = self.x    = np.empty((ts+1, self.N_units+1))
        x[0]            = np.concatenate((x0, self.bias_out))

        # Setting initial reservoir state
        self.psi = Xa[0, :self.N_units]

        # Iterate over time steps
        for i in np.arange(1, ts+1):

            self.bias_in = np.array([np.mean(np.abs((u-self.u_mean)/self.norm))])
            self.step()    # Advance one time step


    def train(self):
        """ Trains the ESN

            Saves:
                Wout: Optimal output matrix
        """

        # Generate the training data
        self.solver.generate(override)

        # Save data to attributes
        U_washout = self.solver.U_washout
        U_train   = self.solver.U_train
        Y_train   = self.solver.Y_train
        u_mean    = self.solver.u_mean
        norm      = self.solver.norm

        N_splits  = self.N_splits

        # To be optimized!
        tikh      = self.tikh
        sigma_in  = self.sigma_in
        rho       = self.rho

        # Output bias
        self.bias_out = np.array([1.])

        # Washout phase
        self.psi = np.zeros(self.N_units)
        self.open_loop(N_splits[0])

        # LHS and RHS are the left and right hand sides of the equation Wout = LHS \ RHS
        LHS   = 0
        RHS   = 0

        # Split the training data into N_splits parts - // is integer division (eg 5.9//2 = 2)
        N_len = (U_train.shape[0]-1)//N_splits

        # Loop over the splits
        for ii in range(N_splits):

            # Open-loop train phase - Xa1 is the augmented reservoir state time series, xf is the final state
            Xa1 = self.open_loop(U_train[ii*N_len:(ii+1)*N_len], xf)[1:]
            xf  = Xa1[-1,:self.N_units].copy()

            LHS += np.dot(Xa1.T, Xa1)                                       # Addding updated reservoir state squared to LHS
            RHS += np.dot(Xa1.T, Y_train[ii*N_len:(ii+1)*N_len])            # Adding updated reservoir state and output to RHS

        # To cover the last part of the data that didn't make into the even splits
        if N_splits > 1:
            Xa1 = self.open_loop(U_train[(ii+1)*N_len:], xf)[1:]
            LHS += np.dot(Xa1.T, Xa1)
            RHS += np.dot(Xa1.T, Y_train[(ii+1)*N_len:])

        LHS.ravel()[::LHS.shape[1]+1] += tikh

        self.W_out = np.linalg.solve(LHS, RHS)


# # Test the ESN with the test data

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