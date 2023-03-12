import numpy as np

# Local imports
from .ode import Solver


class VDP(Solver):
    """ Van der Pol system solver class """

    def __init__(self, params, dt, N_sets, upsample=1, seed=0):
        super().__init__(params, dt, N_sets, upsample, seed)

        self.dim = 2


    def ddt(self, u, *args):
        """ Now for the VdP system """

        mu, omega = self.params
        x, y      = u

        return [y, mu*(1 - x**2)*y - x]


class Slope(Solver):
    """ Literally just slope """

    def __init__(self, params, dt, N_sets, upsample=1, seed=0):
        super().__init__(params, dt, N_sets, upsample, seed)

        self.dim = 1

    def ddt(self, u, *args):
        m1, m2 = self.params

        return [m1]