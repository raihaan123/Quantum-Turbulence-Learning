import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigs
import time

# Local imports
from tools.decorators import debug, hyperparameters
from .RCM import RCM


class CRCM(RCM):
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
                       connectivity=3,
                       sigma_in=1,
                       win_range=(-1, 1),
                       w_range=(-1, 1),
                       activation=np.tanh,
                       eps=1e-2,
                       tik=1e-6,
                       seed=0):

        super().__init__(solver, eps, tik, seed)

        ### Defining attributes of the CRCM ###
        N = self.N_dof      = N_units                   # Number of reservoir units - ie the degree of freedom of the reservoir
        self.N_splits       = N_splits
        self.sigma_in       = sigma_in
        self.connectivity   = connectivity
        self.sparseness     = 1 - connectivity/(N-1)
        self.activation     = activation

        # Defining reservoir state
        self.psi            = np.zeros(N)

        # Initialize the input weight matrix Win
        row_indices = np.arange(N)
        col_indices = self.rnd.randint(0, self.N_in, size=N)
        data = self.rnd.uniform(*win_range, size=N)
        self.Win = csr_matrix((data, (row_indices, col_indices)), shape=(N, self.N_in))

        # On average only connectivity elements different from zero
        self.W = csr_matrix(self.rnd.uniform(*w_range, (N, N)) * (self.rnd.rand(N, N) < (1-self.sparseness)))

        # The spectral radius of W is the maximum absolute value of its eigenvalues
        self.rho = np.abs(eigs(self.W, k=1, which='LM', return_eigenvectors=False))[0]

        # Rescale W to have a spectral radius of 1
        self.W *= 1/self.rho


    def step(self):
        """ Advances one ESN time step """
        # Reservoir update - accessing the current reservoir state from the class attribute
        self.psi    = self.activation(self.Win.dot(self.X) + self.W.dot(self.psi))