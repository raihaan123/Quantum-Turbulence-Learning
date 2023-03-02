"""
    Introduces functions to solve ODEs using the forward Euler method

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib for dark background and latex fonts
plt.style.use('dark_background')
plt.rcParams.update({'text.usetex'      : True,
                     'font.family'      : 'serif',
                     'figure.figsize'   : (15,5),
                     'font.size'        : 20
})

#############################
### The actual functions! ###
#############################

def ddt_lorentz(u, params):
    """
        Returns the time derivative of u - specific to the Lorentz system of ODEs.
    """
    beta, rho, sigma    = params
    x, y, z             = u

    return np.array([sigma*(y-x),
                     x*(rho-z)-y, 
                     x*y-beta*z])



def ddt_MFE():
    None


def forward_euler(ddt, u0, T, *args):
    """
        Forwards Euler method for solving ODEs - coded generically so that it can be used for any ODE system.

        Args:
            ddt: function that returns the time derivative of u
            u0: initial condition
            T: time steps
            *args: additional arguments to ddt

        Returns:
            u: Time series of shape (T.size, u0.size)
    """

    u = np.empty((T.size, u0.size))
    u[0] = u0

    for i in range(1, T.size):
        u[i] = u[i-1] + (T[i] - T[i-1]) * ddt(u[i-1], *args)

    return u


def solve_ode(N, dt, u0, params=[8/3, 28, 10], ddt=ddt_lorentz):
    """
        Solves the ODEs for N time steps starting from u0.
        Returned values are normalized.

        Args:
            N: number of time steps
            dt: time step
            u0: initial condition
            params: parameters of the ODE system
            ddt: function that returns the time derivative of u

        Returns:
            normalized time series of shape (N+1, u0.size)
    """

    T = np.arange(N+1) * dt
    U = forward_euler(ddt, u0, T, params)

    return U


def generate_data(dim, upsample, dt, ddt=ddt_lorentz, noisy=True):
    """
        Generates data for training, validation and testing.
        Args:
            dim: dimension of the ODE system
            upsample: upsampling factor
            dt: time step
            ddt: function that returns the time derivative of u
            noisy: whether to add noise to the data

        Returns:
            U_washout: washout data
            U_train: training + validation data
            Y_train: training + validation data to match at next timestep
            U_test: test data
            Y_test: test data to match at next timestep
            norm: normalisation factor
            u_mean: mean of training data
    """

    # Number of time steps for transient
    N_transient = int(200/dt)

    # Runnning transient solution to reach attractor
    u0   = solve_ode(N_transient, dt/upsample, np.random.random((dim)), ddt=ddt_lorentz)[-1]

    # Lyapunov time and corresponding time steps
    t_lyap      = 0.906**(-1)     # Lyapunov Time (inverse of largest Lyapunov exponent)
    N_lyap      = int(t_lyap/dt)

    # Number of time steps for washout, training, validation and testing
    N = np.hstack((np.array([50]), np.array([50, 3, 1000]) * N_lyap))
    N_washout, N_train, N_val, N_test = N

    # Generate data for training, validation and testing (and washout period)
    U           = solve_ode(sum(N)*upsample, dt/upsample, u0, ddt=ddt)[::upsample]

    # Compute normalization factor (range component-wise)
    U_data      = U[:N_washout+N_train].copy()      # [:x] means from 0 to x-1 --> ie first x elements
    m           = U_data.min(axis=0)                # axis=0 means along columns
    M           = U_data.max(axis=0)
    norm        = M-m
    u_mean      = U_data.mean(axis=0)

    # Washout data
    U_washout   = U[:N_washout].copy()

    # Training + Validation data
    U_train        = U[N_washout       : N_washout+N_train-1].copy() # Inputs
    Y_train        = U[N_washout+1     : N_washout+N_train  ].copy() # Data to match at next timestep

    # Data to be used for testing
    U_test      = U[N_washout+N_train    : N_washout+N_train+N_test-1].copy()
    Y_test      = U[N_washout+N_train+1  : N_washout+N_train+N_test  ].copy()

    # Plotting part of training data to visualize noise
    plt.plot(U_train[:N_val,0], c='w', label='Non-noisy')
    plt.plot(U_train[:N_val], c='w')
    
    # Adding noise to training set inputs with sigma_n the noise of the data
    # improves performance and regularizes the error as a function of the hyperparameters

    # Random seed for reproducing u0 and the data
    seed = 0   
    rnd1  = np.random.RandomState(seed)

    if noisy:
        data_std = np.std(U, axis=0)
        sigma_n = 1e-6     # Controls noise in training inputs (up to 1e-1)
        for i in range(dim):
            U_train[:,i] = U_train[:,i] \
                            + rnd1.normal(0, sigma_n*data_std[i], N_train-1)

        plt.plot(U_train[:N_val,0], 'r--', label='Noisy')
        plt.plot(U_train[:N_val], 'r--')

    plt.legend()
    # plt.show()    # Just temporarily!

    # Data is loaded into dictionary
    data = {'U_washout' : U_washout,
            'U_train'   : U_train,
            'Y_train'   : Y_train,
            'U_test'    : U_test,
            'Y_test'    : Y_test,
            'norm'      : norm,
            'u_mean'    : u_mean}

    return data