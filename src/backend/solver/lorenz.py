import numpy as np

# Local imports
from .ode import Solver


class Lorenz(Solver):
    """ Lorentz system solver class """

    def __init__(self, params, dt, N_sets, upsample=1, autoencoder=None, noise=0, seed=0):
        super().__init__(params, dt, N_sets, upsample, autoencoder, noise, seed)

        self.dim = 3


    def ddt(self, u, *args):
        """ Returns the time derivative of u - specific to the Lorentz system """

        beta, rho, sigma    = self.params
        x, y, z             = u

        return [sigma*(y-x),
                x*(rho-z)-y,
                x*y-beta*z]
