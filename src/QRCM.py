from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
import numpy as np
from numpy import pi
import time

from decorators import debug

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


    def __init__(self, N_dof=5,
                       seed=0):
        """
        Initialize the QRCM.
        
        Here's a brief overview of the dimensions of the matrices and vectors involved in the QRCM:
        -	Input signal X^t: This is a vector of dimension N_dof, representing the input signal to the system at time t
        -	Quantum state |psi^t>: This is a complex vector of dimension 2^n, representing the state of the quantum system at time t
        -	Output weight matrix W_out: This is a matrix of dimension N_output x N_dof, where N_output represents the number of outputs desired from the system. The matrix maps the information stored in the quantum state to the desired output, effectively reducing the high-dimensional information in the quantum state to a lower-dimensional representation that can be used for further analysis or control
        -	Evolved probability vector P_tilde^(t+1): This is a vector of dimension N_dof, representing the probability vector (squared amplitudes) after temporal evolution through the reservoir
        -	Final probability vector P^(t+1): This is a vector of dimension N_dof, representing the final probability vector after the linear combination of the evolved probability vector and the previous probability vector has been performed using the leaking rate epsilon
        """

        # Defining attributes of the QRCM

        self.N_dof      = N_dof                                         # Number of degrees of freedom
        self.N_qubits   = int(np.ceil(np.log2(self.N_dof)))             # Number of qubits required to represent the input signal

        self.seed       = seed                                          # Set the seed for the random number generator
        self.rnd        = np.random.RandomState(self.seed)              # Random state object

        self.X          = np.zeros((self.N_dof))                        # X^t is the latest input signal - this is a vector of dimension N_dof
        self.P          = self.rnd.dirichlet(np.ones(self.N_dof))       # P^t is the probability amplitude vector - this is a real vector of dimension N_dof
        self.beta       = self.rnd.uniform(0, 2*pi, self.N_qubits)      # Beta is random rotation vector - this is a real vector of dimension N_qubits

        self.psi        = np.zeros((2**self.N_qubits, 1))               # |psi^t> is the quantum state - this is a complex vector of dimension 2^n

        self.eps        = 0.1                                           # Leaking rate epsilon -> P^(t+1) = epsilon*P_tilde^(t+1) + (1-epsilon)*P^(t)


    def build_circuit(self):
        """
        Build the quantum circuit!

        Args:
            None

        Saves:
            self.qc (QuantumCircuit): Quantum circuit object

        """

        n = self.N_qubits

        # Define the quantum and classical registers in the circuit
        qr = self.qr    = QuantumRegister(n)
        cr = self.cr    = ClassicalRegister(n)
        self.qc         = QuantumCircuit(qr, cr)

        # Note that the qubits are initialized to |0> so the first unitary transformation is U(4pi*P^t) followed by U(4pi*X^t) and finally U(beta)

        # Loading the reservoir state parameters
        P = self.P
        X = self.X
        b = self.beta

        # Add the unitary matrices
        self.add_U(4 * pi * P)
        self.qc.barrier()
        self.add_U(4 * pi * X)
        self.qc.barrier()
        self.add_U(b)
        
        # Plot the circuit using the built-in plot function - open in a new window (not ipynb!)
        # print("Circuit diagram: ")
        # print(self.qc.draw(output='text'))
        self.qc.draw(output='mpl', filename='QRCM_circuit.png')


    @debug
    def add_U(self, theta):
        """
        Applies a block U(theta) to the quantum circuit

        Args:
            theta (float): Rotation angle vector - size N_qubits (therefore only N_qubits-1 CNOT gates are needed)

        Saves:
            self.qc (QuantumCircuit): Quantum circuit object

        Note from the paper regarding classical data loading:
        "The combination of RY and CNOT gates is continued until the last qubit is reached.
         There, the CNOT is applied to the previous qubit and if not yet finished, the constructor starts at the upper qubit again."
        """
        
        n = self.N_qubits

        # Repeat for each qubit
        for i, angle in enumerate(theta):

            j = i % n           # j is the qubit index and loops from 0 to n-1
            loop = i // n       # loop is the number of times all qubits have been used

            # Apply the RY gate
            self.qc.ry(angle, self.qr[j])
            
            # If not on the last qubit, apply the CNOT gate
            if j < n-1: self.qc.cx(self.qr[j], self.qr[j+1])
            else:       self.qc.cx(self.qr[j], self.qr[j-1])    # As per the note above
                
                
            # Add a barrier if the last qubit has been used (just for visual clarity)
            if j == n-1:  self.qc.barrier()
        

    def open_loop(self):
        """
        Run the QRCM in open-loop mode
        """
        
        None


    def train(self, data):
        """
        Train the QRCM
        """

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