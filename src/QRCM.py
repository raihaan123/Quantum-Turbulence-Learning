from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
import numpy as np
import time

from ode import generate_data

# data = generate_data(dim, upsample, dt, ddt=ddt_lorentz, noisy=True)
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

    - Initialize the qubits to |0>, the input data is given as |x^t>.
    - Apply unitary matrices U(beta), U(4πx^t), U(4πp^t) to the qubits to evolve the state of the reservoir |ψ^(t+1)>.
    - Measure the final state of the reservoir to get the probability vector P^(t+1).
    - Calculate the state of the reservoir X^(t+1) from P^(t+1) using the output weight matrix W_out.
    - Repeat steps 2-4 for T training instances with data instances {x^(t+1), x_tg^(t+1)} where tg is target.
    - Optimize the output weight matrix W_out using the mean squared cost function with a Tikhonov regularization term.
    - The optimized output matrix is given by W_out* = U_tg * R^T (RR^T + beta*I)^-1, where U_tg is a matrix of target outputs and R is a matrix of reservoir states.
    - Once the output weights are optimized and the hyperparameters are tuned, the RCM can run either in the prediction (closed-loop scenario) or reconstruction mode (open-loop scenario).

    Methods:
        - __init__(): Initialize the QRCM.
        - open_loop(): Run the QRCM in open-loop mode.
        - train(): Train the QRCM.
    """

    # Build the quantum circuit
    # Remember that the qubits are initialized to |0> followed by the three unitary matrices U(beta), U(4πx^t), U(4πp^t)
    # p^t is the probability density vector whilst P^(t) is the probability vector - qiskit statevector is a probability vector
    # Note that ode.py has a function that collects the washout, training and test data - use this!

    # Initializing the QRCM
    def __init__(self, N_dof=3):
        """
        Here's a brief overview of the dimensions of the matrices and vectors involved in the QRCM:
        -	Input signal X^t: This is a vector of dimension N_dof, representing the input signal to the system at time t
        -	Quantum state |psi^t>: This is a complex vector of dimension 2^n, representing the state of the quantum system at time t
        -	Probability amplitude vector p^(t+1): This is a real vector of dimension N_dof, representing the squared magnitude of the amplitudes of the quantum state, and providing information about the probability of each state in the quantum system
        -	Output weight matrix W_out: This is a matrix of dimension N_output x N_dof, where N_output represents the number of outputs desired from the system. The matrix maps the information stored in the quantum state to the desired output, effectively reducing the high-dimensional information in the quantum state to a lower-dimensional representation that can be used for further analysis or control
        -	Evolved probability vector p_tilde^(t+1): This is a vector of dimension N_dof, representing the probability vector after it has been evolved through the reservoir
        -	Final probability vector p^(t+1): This is a vector of dimension N_dof, representing the final probability vector after the linear combination of the evolved probability vector and the previous probability vector has been performed using the leaking rate epsilon
        """

        # Number of degrees of freedom
        self.N_dof = N_dof

        # Number of qubits
        self.N_qubits = int(np.log2(self.N_dof))

        # Number of training + test instances
        self.N_washout  = 1000
        self.N_train    = 1000
        self.N_test     = 1000
        self.N_total    = self.N_washout + self.N_train + self.N_test

        # X^t is the latest input signal - this is a vector of dimension N_dof
        self.X = np.zeros((self.N_dof, 1))

        # |psi^t> is the quantum state - this is a complex vector of dimension 2^n
        self.psi = np.zeros((2**self.N_qubits, 1))

        # p^(t+1) is the probability amplitude vector - this is a real vector of dimension N_dof
        self.p = np.zeros((self.N_dof, 1))

    # Build the quantum circuit
    def build_circuit(self):

        # Building the registers and circuit
        n = self.N_qubits

        qr = self.qr = QuantumRegister(n)
        cr = self.cr = ClassicalRegister(n)
        self.qc = QuantumCircuit(qr, cr)
        
        # Add the unitary matrices - note that these are not the actual matrices but the parameters that define them, the data is given as |x^t> in every iteration
        # Note that beta is a random vector of rotation angles as defined in the paper
        # Note that the qubits are initialized to |0> so the first unitary matrix is U(4pi*P^t) followed by U(4pi*X^t) and finally U(beta)

    def add_Ry_Cnot(self, theta):
        """
        Applies a block U(theta) to the quantum circuit 
            - In the form of an RY gate on qubit i followed by a CNOT gate from i to i+1

        Args:
            theta (float): Rotation angle vector - size N_qubits (therefore only N_qubits-1 CNOT gates are needed)
        """
        
        # Repeat for each qubit
        for i, qubit in enumerate(self.qr):
            
            # Apply the RY gate
            self.qc.ry(theta[i], qubit) 
            
            # Apply the CNOT gate except for the last qubit
            if i < self.N_qubits-1:
                self.qc.cx(qubit, self.qr[i+1])


        



    # Run the QRCM in open-loop mode
    def open_loop(self):
        None


    # Train the QRCM
    def train(self):
        None

    

