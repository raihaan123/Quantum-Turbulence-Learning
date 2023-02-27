"""
    Introduces functions to solve ODEs using the forward Euler method.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('dark_background')

#Latex
mpl.rc('text', usetex = True)
mpl.rc('font', family = 'serif')

#############################
### The actual functions! ###
#############################

def ddt_lorentz(u, params):
    """
        Returns the time derivative of u - specific to the Lorentz system of ODEs.
    """
    beta, rho, sigma = params
    x, y, z = u

    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])


def forward_euler(ddt_lorentz, u0, T, *args):
    """
        Forwards Euler method for solving ODEs - coded generically so that it can be used for any ODE system.
    """
    u = np.empty((T.size, u0.size))
    u[0] = u0
    for i in range(1, T.size):
        u[i] = u[i-1] + (T[i] - T[i-1]) * ddt_lorentz(u[i-1], *args)

    return u


def solve_ode(N, dt, u0, params=[8/3, 28, 10]):
    """
        Solves the ODEs for N time steps starting from u0.
        Returned values are normalized.

        Args:
            N: number of time steps
            u0: initial condition
            norm: normalisation factor of u0 (None if not normalised)
            params: parameters for ODE
        Returns:
            normalized time series of shape (N+1, u0.size)
    """

    T = np.arange(N+1) * dt
    U = forward_euler(ddt_lorentz, u0, T, params)

    return U


def generate_data(dim, upsample, dt, noisy=True):
    """
        Generates data for training, validation and testing.
        Returns:
            U_washout: washout data
            U_tv: training + validation data
            Y_tv: training + validation data to match at next timestep
            U_test: test data
            Y_test: test data to match at next timestep
            norm: normalisation factor
            u_mean: mean of training data
    """

    # Number of time steps for transient
    N_transient     = int(200/dt)

    # Runnning transient solution to reach attractor
    u0   = solve_ode(N_transient, dt/upsample, np.random.random((dim)))[-1]

    t_lyap    = 0.906**(-1)     # Lyapunov Time (inverse of largest Lyapunov exponent)
    N_lyap    = int(t_lyap/dt)  # Number of time steps in one Lyapunov time

    # Number of time steps for washout, train, validation, test
    N_washout = 50
    N_train   = 50 * N_lyap
    N_val     = 3 * N_lyap
    N_test    = 1000 * N_lyap

    # Generate data for training, validation and testing (and washout period)
    U         = solve_ode((N_washout+N_train+N_val+N_test)*upsample, dt/upsample, u0)[::upsample]

    # Compute normalization factor (range component-wise)
    U_data = U[:N_washout+N_train].copy()
    m = U_data.min(axis=0)
    M = U_data.max(axis=0)
    norm = M-m
    u_mean = U_data.mean(axis=0)

    # washout
    U_washout = U[:N_washout].copy()

    # data to be used for training + validation
    U_tv  = U[N_washout:N_washout+N_train-1].copy() #inputs
    Y_tv  = U[N_washout+1:N_washout+N_train].copy() #data to match at next timestep

    # data to be used for testing
    U_test = U[N_washout+N_train:N_washout+N_train+N_test-1].copy()
    Y_test = U[N_washout+N_train+1:N_washout+N_train+N_test].copy()

    # plotting part of training data to visualize noise
    plt.rcParams["figure.figsize"] = (15,5)
    plt.rcParams["font.size"] = 20
    plt.plot(U_tv[:N_val,0], c='w', label='Non-noisy')
    plt.plot(U_tv[:N_val], c='w')
    
    # adding noise to training set inputs with sigma_n the noise of the data
    # improves performance and regularizes the error as a function of the hyperparameters

    seed = 0   #to be able to recreate the data, set also seed for initial condition u0
    rnd1  = np.random.RandomState(seed)

    if noisy:
        data_std = np.std(U,axis=0)
        sigma_n = 1e-6     #change this to increase/decrease noise in training inputs (up to 1e-1)
        for i in range(dim):
            U_tv[:,i] = U_tv[:,i] \
                            + rnd1.normal(0, sigma_n*data_std[i], N_train-1)
        plt.plot(U_tv[:N_val,0], 'r--', label='Noisy')
        plt.plot(U_tv[:N_val], 'r--')

    plt.legend()
    plt.show()

    return U_washout, U_tv, Y_tv, U_test, Y_test, norm, u_mean
