import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigs
import time


class CRCM:
    """
    Class for the CRCM - Classical Reservoir Computing Model

    Attributes:
        N_units: number of reservoir units
        N_in: number of input units
        N_out: number of output units
        
        Win: input weights
        W: reservoir weights
        Wout: output weights

    Methods:
        init: initializes the ESN with random Win and W
        step: advances one ESN time step
        open_loop: advances ESN in open-loop
        train: trains the ESN - ie optimizes Wout

    """

    def __init__(self, dim=3,
                       N_units=200,
                       connectivity=3,
                       seed=0):

        self.dim            = dim
        self.N_units        = N_units
        self.connectivity   = connectivity
        self.sparseness     = 1 - connectivity/(N_units-1)

        self.seed           = seed
        self.rnd            = np.random.RandomState(seed)


    def generate_weights(self):
        """
        Generates the input weights and reservoir weights (Win and W)
        """

        N_units     = self.N_units
        dim         = self.dim
        rnd         = self.rnd


        # Sparse syntax for the input matrix
        # Rows correspond to reservoir units/neurons, columns to inputs
        Win = self.Win = lil_matrix((N_units, dim+1))      # +1 for bias input

        # This loop randomly selects a column in Win and fills it with a random number between [-1, 1)
        for i in range(N_units):
            Win[i, rnd.randint(0, dim+1)] = rnd.uniform(-1, 1) # Only one element different from zero

        # Convert to CSR format for faster matrix-vector multiplication
        Win = Win.tocsr()

        # Sparse syntax for the reservoir matrix
        # On average only connectivity elements different from zero
        W = self.W = csr_matrix(
            rnd.uniform(-1, 1, (N_units, N_units)) * (rnd.rand(N_units, N_units) < (1-self.sparseness)))

        # The spectral radius of W is the maximum absolute value of its eigenvalues
        self.rho = np.abs(eigs(W, k=1, which='LM', return_eigenvectors=False))[0]

        # Rescale W to have a spectral radius of 1
        W *= 1/self.rho


    
    def optimize(self):
        None



    def step(self):
        """ Advances one ESN time step

            Args (self):
                x_pre: reservoir state
                u: input
                sigma_in: input scaling
                rho: spectral radius

            Saves:
                x: new reservoir state
        """

        # Load class attributes
        u           = self.u
        sigma_in    = self.sigma_in
        rho         = self.rho

        # Input is normalized and input bias added
        u_augmented = np.hstack(((u-self.u_mean)/self.norm, self.bias_in))

        # Reservoir update - accessing the current reservoir state from the class attribute
        x_post      = np.tanh(self.Win.dot(u_augmented*sigma_in) + self.W.dot(rho*self.x))   

        # Output bias added and state saved
        self.x      = np.concatenate((x_post, self.bias_out))


    def open_loop(self, U, x0):
        """ Advances ESN in open-loop.

            Args:
                U: input time series
                x0: initial reservoir state

            Returns:
                Xa: Time series of augmented reservoir states
        """

        # Output bias
        self.bias_out    = np.array([1.])

        N     = U.shape[0]      # Number of time steps
        Xa    = np.empty((N+1, self.N_units+1))
        Xa[0] = np.concatenate((x0, self.bias_out))

        # Setting initial reservoir state
        self.x = Xa[0, :self.N_units]

        # Iterate over time steps
        for i in np.arange(1, N+1):
            u   = self.u    = U[i-1]
            self.bias_in    = np.array([np.mean(np.abs((u-self.u_mean)/self.norm))])

            # Redundant now but don't want to break anything!
            self.x = Xa[i-1, :self.N_units]

            # Advance one time step
            self.step()

            # Save augmented state
            Xa[i] = self.x

        return Xa


    def train(self, data, tikh=1e-6, sigma_in=1, rho=1, N_splits=4):
        """ Trains the ESN

            Args:
                U_washout: washout input time series
                U_train: training input time series
                Y_train: training output time series
                N_splits: number of splits for training data

                # TODO: The following will be optained from self.optimize() - currently stub

                tikh: Tikhonov factor
                sigma_in: input scaling
                rho: spectral radius

            Saves:
                Wout: Optimal output matrix
        """

        # Generate weights
        self.generate_weights()

        # Save data to attributes
        U_washout = self.U_washout = data['U_washout']
        U_train   = self.U_train   = data['U_train']
        Y_train   = self.Y_train   = data['Y_train']
        u_mean    = self.u_mean    = data['u_mean']
        norm      = self.norm      = data['norm']

        N_splits  = self.N_splits  = N_splits

        # To be optimized!
        tikh      = self.tikh      = tikh
        sigma_in  = self.sigma_in  = sigma_in
        rho       = self.rho       = rho

        # Output bias
        self.bias_out = np.array([1.])

        # Washout phase
        # [-1,:x] indexes the last row of the first x columns
        xf    = self.open_loop(U_washout, np.zeros(self.N_units))[-1,: self.N_units]
        
        # LHS and RHS are the left and right hand sides of the equation Wout = LHS \ RHS
        LHS   = 0
        RHS   = 0

        # Split the training data into N_splits parts - // is integer division (eg 5.9//2 = 2)
        N_len = (U_train.shape[0]-1)//N_splits

        # Loop over the splits
        for ii in range(N_splits):

            # Start timer
            t1  = time.time()

            # Open-loop train phase - Xa1 is the augmented reservoir state time series, xf is the final state
            Xa1 = self.open_loop(U_train[ii*N_len:(ii+1)*N_len], xf)[1:]
            xf  = Xa1[-1,:self.N_units].copy()

            # Stop timer
            t1  = time.time()

            LHS += np.dot(Xa1.T, Xa1) 
            RHS += np.dot(Xa1.T, Y_train[ii*N_len:(ii+1)*N_len])

        if N_splits > 1:# to cover the last part of the data that didn't make into the even splits
            Xa1 = self.open_loop(U_train[(ii+1)*N_len:], xf)[1:]
            LHS += np.dot(Xa1.T, Xa1) 
            RHS += np.dot(Xa1.T, Y_train[(ii+1)*N_len:])

        LHS.ravel()[::LHS.shape[1]+1] += tikh

        Wout = np.linalg.solve(LHS, RHS)

        return Wout