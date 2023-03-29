import numpy as np
import matplotlib.pyplot as plt

# Local imports
from .ode import Solver


class MFE(Solver):
    """ MFE system solver class """

    def __init__(self, params, dt, N_sets, u0=None,
                 upsample=1, autoencoder=None,
                 noise=0, seed=0):

        super().__init__(params, dt, N_sets, u0,
                         upsample, autoencoder,
                         noise, seed)

        self.dim = 9

        self.alpha = 2 * np.pi / params['L_x']
        self.beta = np.pi / 2
        self.gamma = 2 * np.pi / params['L_z']
        self.k1 = np.sqrt(self.alpha**2 + self.gamma**2)
        self.k2 = np.sqrt(self.beta**2 + self.gamma**2)
        self.k3 = np.sqrt(self.alpha**2 + self.beta**2 + self.gamma**2)

        # Extracting parameters
        self.Re = params['Re']
        self.k_l = params['k_l']
        self.k_e = params['k_e']


    def ddt(self, a, *args):
        """ Returns the time derivative of u - specific to the MFE system """

        a1, a2, a3, a4, a5, a6, a7, a8, a9 = a

        # Compute the time derivatives of the coefficients using the governing equations - very messy!

        da1_dt = (self.beta**2 / self.Re - self.beta**2 * a1 / self.Re
                  - np.sqrt(3/2) * self.beta * self.gamma / self.k3 * a6 * a8
                  + np.sqrt(3/2) * self.beta * self.gamma / self.k2 * a2 * a3)

        da2_dt = (-((4 * self.beta**2 / 3) + self.gamma**2) * a2 / self.Re
                  + (5 * np.sqrt(2) / (3 * np.sqrt(3))) * self.gamma**2 / self.k1 * a4 * a6
                  - self.gamma**2 / (np.sqrt(6) * self.k1) * a5 * a7)

        da3_dt = (-((self.beta**2 + self.gamma**2) * a3 / self.Re)
                + (2 / np.sqrt(6)) * self.alpha * self.beta * self.gamma / (self.k1 * self.k2) * (a4 * a7 + a5 * a6)
                + (self.beta**2 * (3 * self.alpha**2 + self.gamma**2) - 3 * self.gamma**2 * (self.alpha**2 + self.gamma**2)) / (np.sqrt(6) * self.k1 * self.k2 * self.k3) * a4 * a8)

        da4_dt = (-((3 * self.alpha**2 + 4 * self.beta**2) * a4 / (3 * self.Re))
                - self.alpha / np.sqrt(6) * a1 * a5
                - (10 / (3 * np.sqrt(6))) * self.alpha**2 / self.k1 * a2 * a6
                - np.sqrt(3/2) * self.alpha * self.beta * self.gamma / (self.k1 * self.k2) * a3 * a7
                - np.sqrt(3/2) * self.alpha**2 * self.beta**2 / (self.k1 * self.k2 * self.k3) * a3 * a8
                - self.alpha / np.sqrt(6) * a5 * a9)

        da5_dt = (-((self.alpha**2 + self.beta**2) * a5 / self.Re)
                + self.alpha / np.sqrt(6) * a1 * a4
                + self.alpha**2 / (np.sqrt(6) * self.k1) * a2 * a7
                - self.alpha * self.beta * self.gamma / (np.sqrt(6) * self.k1 * self.k3) * a2 * a8
                + self.alpha / np.sqrt(6) * a4 * a9
                + (2 / np.sqrt(6)) * self.alpha * self.beta * self.gamma / (self.k1 * self.k2) * a3 * a6)

        da6_dt = (-((3 * self.alpha**2 + 4 * self.beta**2 + 3 * self.gamma**2) * a6 / (3 * self.Re))
                + self.alpha / np.sqrt(6) * a1 * a7
                + np.sqrt(3/2) * self.beta * self.gamma / self.k3 * a1 * a8
                + (10 / (3 * np.sqrt(6))) * (self.alpha**2 - self.gamma**2) / self.k1 * a2 * a4
                - (2 * np.sqrt(2/3)) * self.alpha * self.beta * self.gamma / (self.k1 * self.k2) * a3 * a5
                + self.alpha / np.sqrt(6) * a7 * a9
                + np.sqrt(3/2) * self.beta * self.gamma / self.k3 * a8 * a9)

        da7_dt = (-((self.alpha**2 + self.beta**2 + self.gamma**2) * a7 / self.Re)
                - self.alpha / np.sqrt(6) * (a1 * a6 + a6 * a9)
                + (1 / np.sqrt(6)) * (self.gamma**2 - self.alpha**2) / self.k1 * a2 * a5
                + (1 / np.sqrt(6)) * self.alpha * self.beta * self.gamma / (self.k1 * self.k2) * a3 * a4)

        da8_dt = (-((self.alpha**2 + self.beta**2 + self.gamma**2) * a8 / self.Re)
                + (2 / np.sqrt(6)) * self.alpha * self.beta * self.gamma / (self.k1 * self.k3) * a2 * a5
                + (self.gamma**2 * (3 * self.alpha**2 - self.beta**2 + 3 * self.gamma**2)) / (np.sqrt(6) * self.k1 * self.k2 * self.k3) * a3 * a4)

        da9_dt = (-((9 * self.beta**2) * a9 / self.Re)
                + np.sqrt(3/2) * self.beta * self.gamma / self.k2 * a2 * a3
                - np.sqrt(3/2) * self.beta * self.gamma / self.k3 * a6 * a8)

        return np.array([da1_dt, da2_dt, da3_dt, da4_dt, da5_dt, da6_dt, da7_dt, da8_dt, da9_dt])


    def generate(self):
        self.u0[4] = self.u0[3] + 0.01 * np.random.RandomState(self.seed).rand() # New silly record lol
        super().generate()

        # Finding kinetic energy of the system at each time step
        k = 0.5 * self.U["Train"][:, 1:4] * self.U["Train"][:, 1:4]

        # Kick out laminarized cases (i.e. maximum kinetic energy > threshold, k_l = 0.48



    def plot(self, N_val=None):
        """ Plots data """
        plt.title(f"Training data: {self.__class__.__name__}")

        # Plotting part of training data to visualize noise
        plt.plot(self.ts_train, self.U["Train"][:N_val, 0], c='w', label='Non-noisy')

        plt.savefig(f"..\FYP Logbook\Diagrams\{self.__class__.__name__}_training_data.png")
        plt.show()