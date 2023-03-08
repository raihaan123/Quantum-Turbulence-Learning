import numpy as np
import matplotlib.pyplot as plt

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
        RK4: RK4 time integration
        ddt: time derivative of the ODE system

    Attributes:
        params: parameters for the ODE system
        dt: time step
        N_sets: list of number of time steps for washout, training and testing
        upsample: upsampling factor for data
        noisy: whether the data is noisy or not
    """

    def __init__(self, params, dt, N_sets,
                 upsample=1, autoencoder=None,
                 noisy=False, seed=0):

        self.params     = params
        self.dt         = dt
        self.N_sets     = N_sets
        self.upsample   = upsample
        self.noisy      = noisy

        self.seed       = seed
        self.rnd        = np.random.RandomState(seed)

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


    def RK4(self, N):
        """ RK4 implementation

        Args:
            N: number of time steps

        Returns:
            u: Time series of shape (T.size, u0.size)
        """

        ddt = self.ddt
        dt  = self.dt

        T = np.arange(N+1) * dt
        u = self.u = np.empty((T.size, self.dim))

        u[0] = self.u0

        for i in range(1, T.size):
            k1 = ddt(u[i-1])
            k2 = ddt(u[i-1] + k1*dt/2)
            k3 = ddt(u[i-1] + k2*dt/2)
            k4 = ddt(u[i-1] + k3*dt)

            u[i] = u[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


    def generate(self, override=False):
        """
            Generates data for training, validation and testing.

            Returns:
                U_washout: washout input
                U_train: training input
                Y_train: training output to match at next timestep
                U_test: test input
                Y_test: test output to match at next timestep
                norm: normalisation factor
                u_mean: mean of training data
        """

        dt          = self.dt

        # Number of time steps for transient
        N_transient = int(200/self.dt)

        # Runnning transient solution to reach attractor
        self.u0 = self.rnd.random((self.dim))
        self.RK4(N_transient)
        self.u0 = self.u[-1]

        # Lyapunov time and corresponding time steps
        t_lyap      = 0.906**(-1)     # Lyapunov Time (inverse of largest Lyapunov exponent)
        N_lyap      = int(t_lyap/dt)

        # Number of time steps for washout, training, validation and testing
        N_sets      = self.N_sets

        if not override:
            N_sets  = np.hstack((np.array([N_sets[0]]), np.array([N_sets[1], N_sets[2]]) * N_lyap))

        N_washout, N_train, N_test = N_sets

        # Generate data for training, validation and testing (and washout period)
        self.RK4(sum(N_sets))

        # Compute normalization factor (range component-wise)
        U_data      = self.u[:N_washout+N_train]    # [:x] means from 0 to x-1 --> ie first x elements
        m           = U_data.min(axis=0)            # axis=0 means along columns
        M           = U_data.max(axis=0)
        self.norm   = M-m
        self.u_mean = U_data.mean(axis=0)

        # Saving data
        self.U["Washout"] = self.u[:N_washout]
        self.U["Train"]   = self.u[N_washout    : N_washout+N_train-1]      # Inputs
        self.Y["Train"]   = self.u[N_washout+1  : N_washout+N_train  ]      # Data to match at next timestep

        # Testing data
        self.U["Test"] = self.u[N_washout+N_train    : N_washout+N_train+N_test-1]
        self.Y["Test"] = self.u[N_washout+N_train+1  : N_washout+N_train+N_test  ]

        # Adding noise to training set inputs with sigma_n the noise of the data
        # improves performance and regularizes the error as a function of the hyperparameters

        if self.noisy:
            data_std = np.std(self.u, axis=0)
            sigma_n = 1e-6     # Controls noise in training inputs (up to 1e-1)
            for i in range(self.dim):
                self.U["Train"][:,i] = self.U["Train"][:,i] \
                                + self.rnd.normal(0, sigma_n*data_std[i], N_train-1)

        if self.ae is not None:    self.autoencode()


    def autoencode(self):
        """ Reduces dimensionality of data to latent space """
        None


    def plot(self, N_val=100):
        """ Plots data """

        # Plotting part of training data to visualize noise
        plt.plot(self.U["Train"][:N_val,0], c='w', label='Non-noisy')
        plt.plot(self.U["Train"][:N_val], c='w')

        if self.noisy:
            plt.plot(self.U["Train"][:N_val,0], 'r--', label='Noisy')
            plt.plot(self.U["Train"][:N_val], 'r--')

        plt.legend()
        plt.show()