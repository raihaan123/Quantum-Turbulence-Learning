import numpy as np

# Local imports
from .ode import Solver


# NARMA5 system - with alpha = 0.3, beta = 0.05, gamma = 1.5, delta = 0.1, mu = 0.1
# f0 = 2.11, f1 = 3.73, f2 = 4.11, T=100
class NARMA5(Solver):
    """ NARMA5 system solver class """

    
        
