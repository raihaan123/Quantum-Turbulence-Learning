from .ode import Solver

class MFE(Solver):
    """ MFE system solver class
    
    """

    def __init__(self, params, dt, N_sets, upsample=1, seed=0):
        super().__init__(params, dt, N_sets, upsample=1, seed=0)

        self.dim = 9


    def ddt(self, u):
        """
        Returns the time derivative of u - specific to the MFE system
        """

        None
