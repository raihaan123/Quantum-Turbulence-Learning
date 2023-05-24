import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# Local imports
from tools.decorators import debug

# Configure matplotlib for dark background and latex fonts
plt.style.use('dark_background')
plt.rcParams.update({'text.usetex'      : True,
                     'font.family'      : 'serif',
                     'figure.figsize'   : (15,5),
                     'font.size'        : 20
})


class Solver:
    """ Base numerical solver class for general ODE systems

    Heavily adapted from Alberto Racca's implementation: https://www.sciencedirect.com/science/article/pii/S0893608021001969

    Methods:
        ddt: time derivative of the ODE system
        generate: generates data for washout, training and testing
        plot: plots the data


    Attributes:
        params: parameters for the ODE system
        dt: time step
        N_sets: list of number of time steps for washout, training and testing
        upsample: upsampling factor for data
        noisy: whether the data is noisy or not
    """

    def __init__(self, params, dt, N_sets, u0,
                 upsample, autoencoder,
                 noise, seed):

        self.params     = params
        self.dt         = dt
        self.N_sets     = N_sets
        self.u0         = u0
        self.upsample   = upsample
        self.noise      = noise         # Controls noise in training inputs (up to 1e-1)
        self.seed       = seed

        self.dim        = 0
        self.u          = []

        # Data containers
        self.U          = {"Washout": [],
                           "Train"  : [],
                           "Test"   : []}

        self.Y          = {"Train"  : [],
                           "Test"   : []}

        self.ae         = autoencoder


    def ddt(self):
        raise NotImplementedError


    def generate(self):
        """ Generates data for training, validation and testing.

        Returns:
            U_washout: washout input
            U_train: training input
            Y_train: training output to match at next timestep
            U_test: test input
            Y_test: test output to match at next timestep
            norm: normalisation factor
            u_mean: mean of training data
        """
        # Refresh random generator - in case of calls from multiple RCM instances
        rnd = self.rnd = np.random.RandomState(self.seed)
        dt  = self.dt

        # Initial condition
        if self.u0 is None:
            self.u0 = rnd.random((self.dim))

        # TODO: Implement Lyapunov time function with QR algorithm
        # # Lyapunov time and corresponding time steps
        # t_lyap      = 0.906**(-1)     # Lyapunov Time (inverse of largest Lyapunov exponent)
        # N_lyap      = int(t_lyap/dt)

        # if not override:
        #     N_sets  = np.hstack((np.array([N_sets[0]]), np.array([N_sets[1], N_sets[2]]) * N_lyap))

        N_trans, N_washout, N_train, N_test = self.N_sets

        # Time vector for training and testing plots
        T = np.arange(sum(self.N_sets)) * dt
        self.ts_train = T[N_trans+N_washout: N_trans+N_washout+N_train] - T[N_trans+N_washout]
        self.ts_test  = T[N_trans+N_washout+N_train+1: N_trans+N_washout+N_train+N_test] - T[N_trans+N_washout]

        # Generate data for washout, training and testing using scipy.integrate.odeint
        self.u = odeint(self.ddt, self.u0, T)

        # Compute normalization factor (range component-wise)
        U_data      = self.u[:N_washout+N_train]    # [:x] means from 0 to x-1 --> ie first x elements
        m           = U_data.min(axis=0)            # axis=0 means along columns
        M           = U_data.max(axis=0)

        std         = self.std = U_data.std(axis=0)
        u_mean      = self.u_mean = U_data.mean(axis=0)

        # Saving data
        self.U["Washout"] = self.u[N_trans:N_trans+N_washout]
        self.U["Train"]   = self.u[N_trans+N_washout    : N_trans+N_washout+N_train]      # Inputs
        self.Y["Train"]   = self.u[N_trans+N_washout+1  : N_trans+N_washout+N_train+1]      # Data to match at next timestep

        # Testing data
        self.U["Test"] = self.u[N_trans+N_washout+N_train    : N_trans+N_washout+N_train+N_test-1]
        self.Y["Test"] = self.u[N_trans+N_washout+N_train+1  : N_trans+N_washout+N_train+N_test  ]

        # Adding noise to training set inputs with sigma_n the noise of the data
        # improves performance and regularizes the error as a function of the hyperparameters

        if self.noise != 0:
            data_std = np.std(self.u, axis=0)
            for i in range(self.dim):
                self.U["Train"][:,i] = self.U["Train"][:,i] \
                                + rnd.normal(0, self.noise*data_std[i], N_train)

        # # if self.normalize:
        self.U["Train"] = (self.U["Train"] - u_mean) / std
        self.Y["Train"] = (self.Y["Train"] - u_mean) / std

        self.U["Test"] = (self.U["Test"] - u_mean) / std
        self.Y["Test"] = (self.Y["Test"] - u_mean) / std

        if self.ae is not None:    self.autoencode()


    def autoencode(self):
        """ Reduces dimensionality of data to latent space """
        raise NotImplementedError


    def plot(self, N_val=None):
        """ Plots data """
        plt.title(f"Training data: {self.__class__.__name__}")

        # Plotting part of training data to visualize noise
        plt.plot(self.ts_train, self.U["Train"][:N_val,0], c='w', label='Non-noisy')
        plt.plot(self.ts_train, self.U["Train"][:N_val], c='w')

        if self.noise != 0:
            plt.plot(self.ts_train, self.U["Train"][:N_val,0], 'r--', label='Noisy')
            plt.plot(self.ts_train, self.U["Train"][:N_val], 'r--')
            plt.legend()

        plt.savefig(f"..\FYP Logbook\Diagrams\{self.__class__.__name__}_training_data.png")
        plt.show()
