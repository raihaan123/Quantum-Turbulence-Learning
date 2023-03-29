import numpy as np

# Local imports
from .ode import Solver


class Lorenz(Solver):
    """ Lorentz system solver class """

    def __init__(self, params, dt, N_sets, u0=None,
                 upsample=1, autoencoder=None,
                 noise=0, seed=0):

        super().__init__(params, dt, N_sets, u0,
                         upsample, autoencoder,
                         noise, seed)

        self.dim = 3

        # Extracting parameters
        self.beta    = self.params['beta']
        self.rho     = self.params['rho']
        self.sigma   = self.params['sigma']


    def ddt(self, u, *args):
        """ Returns the time derivative of u - specific to the Lorentz system """

        x, y, z             = u

        return [self.sigma*(y-x),
                x*(self.rho-z)-y,
                x*y-self.beta*z]
