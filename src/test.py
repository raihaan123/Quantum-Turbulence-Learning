'''
We investigate 2D turbulence governed by the incompressible Navier-Stokes equations

(1) ∇ · u = 0
(2) ∂_t{u} + u · ∇u = −∇p + 1/Re ∆u + f

where u = (u, v) is the velocity field, p is the pressure, Re is the Reynolds number, and f is a harmonic volume force defined as f = (sin(k_f*y), 0) in cartesian coordinates.

The Navier-Stokes equations are solved on a domain Ω ≡ [0, 2π] × [0, 2π] with periodic boundary conditions.
The solution of this problem is also known as the 2D Kolmogorov flow.

The flow has a laminar solution u = Re*k_f^(-2)*sin(k_f*y), v = 0, which is unstable for sufficiently large Reynolds numbers and wave numbers k_f.

Here, we take kf = 4 and Re = 30 to guarantee the development of a turbulent solution.

The set of Equations above are solved on a uniform N × N grid, with N = 24, using a pseudo-spectral
code with explicit Euler in time with a timestep, ∆t = 0.01, to ensure numerical stability.

'''

# The 2D Kolmogorov flow is a type of 2D turbulence.

# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from scipy.fftpack import fft2, ifft2, fftfreq
import scipy

# Set the parameters of the simulation
N = 24
L = 2*np.pi
dx = L/N
x = np.arange(0, L, dx)
y = np.arange(0, L, dx)
X, Y = np.meshgrid(x, y)
dt = 0.01
t = 0
Re = 30
kf = 4
f = np.zeros((N, N))
f[:, int(N/2)] = np.sin(kf*y)