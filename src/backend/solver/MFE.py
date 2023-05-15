import numpy as np
import matplotlib.pyplot as plt

# Local imports
from .ode import Solver


class MFE(Solver):
    """ MFE system solver class """

    def __init__(self, params, dt, N_sets, u0=None, upsample=1, autoencoder=None,
                 noise=0, seed=0, domain=[[0,10],[0,10],[0,10]], n_points=200):

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

        # Generate the eigenmodes
        # self.generate_eigenmodes(domain, n_points)


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
        plt.plot(self.ts_train, self.U["Train"][:N_val, :], c='w', label='Non-noisy')

        plt.savefig(f"..\FYP Logbook\Diagrams\{self.__class__.__name__}_training_data.png")
        plt.show()


    def generate_eigenmodes(self, domain, n_points):
        """ Generates the eigenmodes of the MFE system

        Parameters:
                domain = [[xi xf], ...] - the domain of each dimension
                n_points - number of points in each dimension
        """

        # Data is the magnitudes of each eigenmode - N in total, with T timesteps - so data is [T x N]
        # We use the eigenmodes to reconstruct the data in the original space - equations as follows:

        # Create the grid
        x = np.linspace(domain[0][0], domain[0][1], n_points)
        y = np.linspace(domain[1][0], domain[1][1], n_points)
        z = np.linspace(domain[2][0], domain[2][1], n_points)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        alpha = self.alpha
        gamma = self.gamma

        # Calculate the velocity profiles for each mode
        V1 = np.array([np.sqrt(2) * np.sin(np.pi * Y / 2), np.zeros_like(Y), np.zeros_like(Y)])
        V2 = np.array([4/np.sqrt(3) * np.cos(np.pi * Y / 2)**2 * np.cos(gamma * Z), np.zeros_like(Y), np.zeros_like(Y)])
        V3 = np.array([np.zeros_like(Y), 2 * gamma * np.cos(np.pi * Y / 2) * np.cos(gamma * Z), np.pi * np.sin(np.pi * Y / 2) * np.sin(gamma * Z)]) * (2 / np.sqrt(4 * gamma**2 + np.pi**2))
        V4 = np.array([np.zeros_like(Y), np.zeros_like(Y), 4/np.sqrt(3) * np.cos(alpha * X) * np.cos(np.pi * Y / 2)**2])
        V5 = np.array([np.zeros_like(Y), np.zeros_like(Y), 2 * np.sin(alpha * X) * np.sin(np.pi * Y / 2)])
        V6 = np.array([-gamma * np.cos(alpha * X) * np.cos(np.pi * Y / 2)**2 * np.sin(gamma * Z), np.zeros_like(Y), alpha * np.sin(alpha * X) * np.cos(np.pi * Y / 2)**2 * np.cos(gamma * Z)]) * (4 * np.sqrt(2) / np.sqrt(3 * (alpha**2 + gamma**2)))
        V7 = np.array([gamma * np.sin(alpha * X) * np.sin(np.pi * Y / 2) * np.sin(gamma * Z), np.zeros_like(Y), alpha * np.cos(alpha * X) * np.sin(np.pi * Y / 2) * np.cos(gamma * Z)]) * (2 * np.sqrt(2) / np.sqrt(alpha**2 + gamma**2))
        N_8 = 2 * np.sqrt(2) / np.sqrt((alpha**2 + gamma**2) * (4 * alpha**2 + 4 * gamma**2 + np.pi**2))
        V8 = np.array([np.pi * alpha * np.sin(alpha * X) * np.sin(np.pi * Y / 2) * np.sin(gamma * Z), 2 * (alpha**2 + gamma**2) * np.cos(alpha * X) * np.cos(np.pi * Y / 2) * np.sin(gamma * Z), -np.pi * gamma * np.cos(alpha * X) * np.sin(np.pi * Y / 2) * np.cos(gamma * Z)]) * N_8
        V9 = np.array([np.sqrt(2) * np.sin(3 * np.pi * Y / 2), np.zeros_like(Y), np.zeros_like(Y)])

#         self.plot_heatmap(V7, n_points)


#     def plot_heatmap(self, V, n_points, cmap='viridis'):
#         """ Create a heatmap for the magnitude of the velocity field for a given mode at a specific z slice.

#         Parameters:
#         - V: the velocity field for the mode (a 4D array)
#         - z_slice: the index of the z slice to visualize (default is the middle slice)
#         - cmap: the colormap to use (default is 'viridis')
#         """
#         # Calculate the magnitude of the velocity
#         magnitude = np.sqrt(np.sum(V**2, axis=0))
#         select=n_points//2

#         # Create the heatmap
#         plt.figure(figsize=(8, 6))
#         plt.imshow(magnitude[:, :, select], cmap=cmap, origin='lower')
#         plt.colorbar(label='Velocity magnitude')
#         plt.show()