from .ode import Solver

import numpy as np


class Lorentz(Solver):
    """ Lorentz system solver class
    
    """

    def __init__(self, params, dt, N_sets, upsample=1, seed=0):
        super().__init__(params, dt, N_sets, upsample=1, seed=0)

        self.dim = 3


    def ddt(self, u):
        """
        Returns the time derivative of u - specific to the Lorentz system
        """
        beta, rho, sigma    = self.params
        x, y, z             = u

        return np.array([sigma*(y-x),
                        x*(rho-z)-y,
                        x*y-beta*z])
