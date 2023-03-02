from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
import numpy as np
from numpy import pi
import time

# data = generate_data(dim, upsample, dt, ddt=ddt_lorentz, noisy=True) --> return type is dictionary
#
# data : {'U_washout' : U_washout,
#         'U_train'   : U_train,
#         'Y_train'   : Y_train,
#         'U_test'    : U_test,
#         'Y_test'    : Y_test,
#         'norm'      : norm,
#         'u_mean'    : u_mean}

class QRCM:
    """
    Class for the QRCM - Quantum Reservoir Computing Model

    1) Intialize the qubits to |0>, the input data is given as |x^t>.
    2) Apply unitary matrices U(beta), U(4πx^t), U(4πp^t) to the qubits to evolve the state of the reservoir |ψ^(t+1)>.
    3) Measure the final state of the reservoir to get the probability vector P^(t+1).
    4) Calculate the state of the reservoir X^(t+1) from P^(t+1) using the output weight matrix W_out.
    5) Repeat steps 2-4 for T training instances with data instances {x^(t+1), x_tg^(t+1)} where tg is target.
    6) Optimize the output weight matrix W_out using the mean squared cost function with a Tikhonov regularization term.
    7) The optimized output matrix is given by W_out* = U_tg * R^T (RR^T + beta*I)^-1, where U_tg is a matrix of target outputs and R is a matrix of reservoir states.
    8) Once the output weights are optimized and the hyperparameters are tuned, the RCM can run either in the prediction (closed-loop scenario) or reconstruction mode (open-loop scenario).


    Methods:
        - __init__(): Initialize the QRCM.
        - open_loop(): Run the QRCM in open-loop mode.
        - train(): Train the QRCM.
    """

    # Build the quantum circuit
    # Remember that the qubits are initialized to |0> followed by the three unitary matrices U(beta), U(4πx^t), U(4πp^t)
    # p^t is the probability density vector whilst P^(t) is the probability vector - qiskit statevector is a probability vector

    # Initializing the QRCM
    def __init__(self, N_dof=3,
                       seed=0):
        """
        Here's a brief overview of the dimensions of the matrices and vectors involved in the QRCM:
        -	Input signal X^t: This is a vector of dimension N_dof, representing the input signal to the system at time t
        -	Quantum state |psi^t>: This is a complex vector of dimension 2^n, representing the state of the quantum system at time t
        -	Probability amplitude vector p^(t+1): This is a real vector of dimension N_dof, representing the squared magnitude of the amplitudes of the quantum state, and providing information about the probability of each state in the quantum system
        -	Output weight matrix W_out: This is a matrix of dimension N_output x N_dof, where N_output represents the number of outputs desired from the system. The matrix maps the information stored in the quantum state to the desired output, effectively reducing the high-dimensional information in the quantum state to a lower-dimensional representation that can be used for further analysis or control
        -	Evolved probability vector P_tilde^(t+1): This is a vector of dimension N_dof, representing the probability vector after it has been evolved through the reservoir
        -	Final probability vector P^(t+1): This is a vector of dimension N_dof, representing the final probability vector after the linear combination of the evolved probability vector and the previous probability vector has been performed using the leaking rate epsilon
        """

        # Defining attributes of the QRCM

        self.N_dof      = N_dof                                     # Number of degrees of freedom
        self.N_qubits   = int(np.log2(self.N_dof))                  # Number of qubits

        self.seed       = seed                                      # Set the seed for the random number generator
        self.rnd        = np.random.RandomState(self.seed)          # Random state object

        self.X          = np.zeros((self.N_dof, 1))                 # X^t is the latest input signal - this is a vector of dimension N_dof
        self.P          = np.random.dirichlet(self.N_dof)           # P^t is the probability amplitude vector - this is a real vector of dimension N_dof
        self.beta       = self.rnd.uniform(0, 2*pi, self.N_qubits)  # Beta is random rotation vector - this is a real vector of dimension N_qubits

        self.psi        = np.zeros((2**self.N_qubits, 1))           # |psi^t> is the quantum state - this is a complex vector of dimension 2^n

        self.eps        = 0.1                                       # Leaking rate epsilon -> P^(t+1) = epsilon*P_tilde^(t+1) + (1-epsilon)*P^(t)

        ### Archive ###
        # self.p = self.rnd.uniform(0, 1, self.N_dof)
        # self.p /= np.linalg.norm(self.p)
        # self.P = self.p**2

    def build_circuit(self):
        """
        Build the quantum circuit!

        Args:
            None

        Saves:
            self.qc (QuantumCircuit): Quantum circuit object

        """

        n = self.N_qubits

        qr = self.qr    = QuantumRegister(n)
        cr = self.cr    = ClassicalRegister(n)
        self.qc         = QuantumCircuit(qr, cr)

        # Note that beta is a random vector of rotation angles as defined in the paper
        # Note that the qubits are initialized to |0> so the first unitary matrix is U(4pi*P^t) followed by U(4pi*X^t) and finally U(beta)

        # Loading the reservoir state parameters
        P = self.P
        X = self.X
        b = self.beta

        # Add the unitary matrices
        self.add_U(4 * pi .* P)
        self.add_U(4 * pi .* X)
        self.add_U(b)


    def add_U(self, theta):
        """
        Applies a block U(theta) to the quantum circuit 
            - In the form of an RY gate on qubit i followed by a CNOT gate from i to i+1

        Args:
            theta (float): Rotation angle vector - size N_qubits (therefore only N_qubits-1 CNOT gates are needed)

        Saves:
            self.qc (QuantumCircuit): Quantum circuit object
            
        Note from the paper:
        "The combination of RY and CNOT gates is continued until the last qubit is reached.
         There, the CNOT is applied to the previous qubit and if not yet finished, the constructor starts at the upper qubit again."
        """

        # Repeat for each qubit
        for i, angle in enumerate(theta):

            # If on the last qubit, reset i to 0
            j = i % self.N_qubits-1     # -1 because we want to start at 0

            # Apply the RY and CNOT gates
            self.qc.ry(angle, self.qr[j]) 
            self.qc.cx(self.qr[j], self.qr[j+1])


    # Run the QRCM in open-loop mode
    def open_loop(self):
        None


    # Train the QRCM
    def train(self, data):

        # Save data to attributes
        U_washout = self.U_washout = data['U_washout']
        U_train   = self.U_train   = data['U_train']
        Y_train   = self.Y_train   = data['Y_train']
        u_mean    = self.u_mean    = data['u_mean']
        norm      = self.norm      = data['norm']

        # Initialize the output weight matrix randomly
        self.W_out = np.zeros((self.N_dof, self.N_dof))

        # Initialize the reservoir state matrix
        self.R = np.zeros((self.N_dof, self.N_dof))

        # Initialize the target output matrix
        self.U_tg = np.zeros((self.N_dof, self.N_dof))