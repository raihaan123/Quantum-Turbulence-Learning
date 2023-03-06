import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

    def __init__(self, params, dt, N_sets, upsample=1, noisy=False, seed=0):
        self.params     = params
        self.dt         = dt
        self.N_sets     = N_sets
        self.upsample   = upsample
        self.noisy      = noisy

        self.seed       = seed
        self.rnd        = np.random.RandomState(seed)
        
        self.dim        = 0
        self.u0         = []
        self.u          = []
        self.data       = {}


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
                U_washout: washout data
                U_train: training data
                Y_train: training data to match at next timestep
                U_test: test data
                Y_test: test data to match at next timestep
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
            N_sets = np.hstack((np.array([N_sets[0]]), np.array([N_sets[1], N_sets[2]]) * N_lyap))

        N_washout, N_train, N_test = N_sets

        # Generate data for training, validation and testing (and washout period)
        self.RK4(sum(N_sets))

        # Compute normalization factor (range component-wise)
        U_data      = self.u[:N_washout+N_train].copy()      # [:x] means from 0 to x-1 --> ie first x elements
        m           = U_data.min(axis=0)                # axis=0 means along columns
        M           = U_data.max(axis=0)
        norm        = M-m
        u_mean      = U_data.mean(axis=0)

        # Washout data
        U_washout   = self.u[:N_washout].copy()
        U_train     = self.u[N_washout       : N_washout+N_train-1].copy() # Inputs
        Y_train     = self.u[N_washout+1     : N_washout+N_train  ].copy() # Data to match at next timestep

        # Data to be used for testing
        U_test      = self.u[N_washout+N_train    : N_washout+N_train+N_test-1].copy()
        Y_test      = self.u[N_washout+N_train+1  : N_washout+N_train+N_test  ].copy()

        # Plotting part of training data to visualize noise
        # plt.plot(U_train[:N_val,0], c='w', label='Non-noisy')
        # plt.plot(U_train[:N_val], c='w')
        
        # Adding noise to training set inputs with sigma_n the noise of the data
        # improves performance and regularizes the error as a function of the hyperparameters

        if self.noisy:
            data_std = np.std(U, axis=0)
            sigma_n = 1e-6     # Controls noise in training inputs (up to 1e-1)
            for i in range(self.dim):
                U_train[:,i] = U_train[:,i] \
                                + self.rnd.normal(0, sigma_n*data_std[i], N_train-1)

        #     plt.plot(U_train[:N_val,0], 'r--', label='Noisy')
        #     plt.plot(U_train[:N_val], 'r--')

        # plt.legend()
        # plt.show()    # Just temporarily!

        # Data is loaded into dictionary
        self.data = {   'U_washout' : U_washout,
                        'U_train'   : U_train,
                        'Y_train'   : Y_train,
                        'U_test'    : U_test,
                        'Y_test'    : Y_test,
                        'norm'      : norm,
                        'u_mean'    : u_mean}


### TESTING THE DATA GENERATION ###
# Data generation parameters
# dim             = 3
# upsample        = 2                     # To increase the dt of the ESN wrt the numerical integrator
# dt              = 0.005 * upsample      # Time step
# N_sets          = [50, 50, 1000]       # Washout, training, validation, testing

# data = generate_data(dim, N_sets, upsample, dt, ddt=ddt_lorentz, noisy=True)

# # Print data keys and shapes
# [print(key, value.shape) for key, value in data.items()]

# norm [34.93270814 45.44324541 35.74071098]
# u_mean [-0.53059272 -0.50746428 24.16344666]