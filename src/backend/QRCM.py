from qiskit import QuantumCircuit, QuantumRegister, execute, Aer
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, tanh
from tqdm import tqdm

# Local imports
from tools.decorators import debug, hyperparameters
from .RCM import RCM


class ProbabilityVectorError(Exception):
    """ Raised when the probability vector is not normalized """
    pass


class QRCM(RCM):
    """ Class for the QRCM - Quantum Reservoir Computing Model

    My implementation based on "Hybrid quantum-classical reservoir computing of thermal convection flow": https://ar5iv.labs.arxiv.org/html/2204.13951

    1) Intialize the qubits to |0>, the input data is given as |x^t>.
    2) Apply unitary matrices U(beta), U(4πx^t), U(4πp^t) to the qubits to evolve the state of the reservoir |ψ^(t+1)> - retrived using statevector simulator.
    3) Square the statevector to get the probability vector P_tilde^(t+1). Apply the leaking rate epsilon to get the final probability vector P^(t+1).
    4) Calculate the state of the dynamical system X^(t+1) from P^(t+1) using the output weight matrix W_out.
    5) Repeat steps 2-4 for T training instances with data pairs {x^(t+1), x_tg^(t+1)} where tg is target output.
    6) The optimized output matrix is given by W_out* = U_tg * R^T (RR^T + beta*I)^-1, where U_tg is a matrix of target outputs and R is a matrix of reservoir states.
    7) Once the output weights are optimized and the hyperparameters are tuned, the RCM can run either in the prediction (closed-loop) or reconstruction mode (open-loop).


    Methods:
        - __init__(): Initialize the QRCM.
        - open_loop(): Run the QRCM in open-loop mode.
        - train(): Train the QRCM.
    """

    def __init__(self,  solver=None,
                        qubits=2,
                        eps=1e-2,
                        tik=1e-2,
                        seed=0,
                        plot=False):
        """ Initialize the QRCM

        Here's a brief overview of the dimensions of the matrices and vectors involved in the QRCM:
        -	Input signal X^t: This is a vector of dimension N_in, representing the input signal to the system at time t
        -	Quantum state |psi^t>: This is a complex vector of dimension 2^n=N_dof, representing the state of the quantum system at time t
        -	Output weight matrix W_out: This is a matrix of dimension N_output x N_dof, where N_output represents the number of outputs desired from the system. The matrix maps the information stored in the quantum state to the desired output, effectively reducing the high-dimensional information in the quantum state to a lower-dimensional representation that can be used for further analysis or control
        -	Evolved probability vector P_tilde^(t+1): This is a vector of dimension N_dof, representing the probability vector (squared amplitudes) after temporal evolution through the reservoir
        -	Final probability vector P^(t+1): This is a vector of dimension N_dof, representing the final probability vector after the linear combination of the evolved probability vector and the previous probability vector has been performed using the leaking rate epsilon
        """

        super().__init__(solver, eps, tik, seed)

        ### Defining attributes of the QRCM ###
        n  = self.N_qubits  = qubits                                # int(np.ceil(np.log2(dim))) for minimum number of qubits required
        N  = self.N_dof     = 2**n                                  # Number of degrees of freedom of quantum reservoir

        self.qr             = QuantumRegister(n)                    # Define the quantum registers in the circuit
        self.qc             = QuantumCircuit(self.qr)               # Note that the qubits are initialized to |0> by default
        self.plot           = plot                                  # Plot the Quantum Circuit

        self.psi            = np.zeros((N))                         # |psi^t> is the quantum state - this is a complex vector of dimension 2^n
        self.P              = self.rnd.dirichlet(np.ones(N))        # P^t is the probability amplitude vector - this is a real vector of dimension N_dof
        self.beta           = self.rnd.uniform(0, 2*pi, n)          # Beta is random rotation vector - this is a real vector of dimension n

        self.backend = Aer.get_backend('statevector_simulator')     # Aer statevector simulator backend


    def refresh(self):
        """ Reset the reservoir state to initial state """
        super().refresh()

        self.P = self.rnd.dirichlet(np.ones(self.N_dof))




    def Unitary(self, theta, name='theta'):
        """ Applies a block U(theta) to the quantum circuit

        Note from the paper regarding classical data loading:
          "The combination of RY and CNOT gates is continued until
           the last qubit is reached. There, the CNOT is applied
           to the previous qubit and if not yet finished, the
           constructor starts at the upper qubit again."

        Args:
            theta (float): Rotation angle vector - size N_qubits (therefore only N_qubits-1 CNOT gates are needed)

        Saves:
            self.qc (QuantumCircuit): Quantum circuit object
        """

        n = self.N_qubits

        # Repeat for each qubit
        for i, angle in enumerate(theta):
            # j is the qubit index and loops from 0 to n-1
            j = i % n

            # Apply the RY gate
            self.qc.ry(angle, j, label=f'$R_Y$({name})')

            # If not on the last qubit, apply the CNOT gate
            if j < n-1: self.qc.cx(j, j+1)
            else:       self.qc.cx(j, j-1)    # As per the note above

            # Add a barrier if the last qubit has been used (just for visual clarity)
            if j == n-1:  self.qc.barrier()


    def step(self):
        # Reset the previous circuit - create a new instance of QuantumCircuit
        self.qc = QuantumCircuit(self.qr)

        # Loading the reservoir state parameters
        P = self.P
        X = self.X
        b = self.beta

        # Add the unitary transformations to the circuit separated by barriers
        # The first unitary is U(4pi*P^t) followed by U(4pi*X^t) and finally U(beta)
        self.Unitary(10*P, 'P')
        self.qc.barrier()
        self.Unitary(0.01*X, 'X')
        self.qc.barrier()
        self.Unitary(0.01*b, 'b')

        # Run the circuit - find state probability vector using statevector_simulator
        psi     = self.psi      = np.abs(execute(self.qc, self.backend).result().get_statevector())
        P_tilde = self.P_tilde  = np.abs(psi)**2

        # Solve for the final probability vector P^(t+1)
        P = self.eps * P_tilde + (1-self.eps) * P
        if not np.isclose(np.sum(P), 1):
            raise ProbabilityVectorError("Probability vector is not valid!")


    # Cheeky wrapper of the open_loop method to plot the circuit at the end
    def open_loop(self, key, save=False):
        super().open_loop(key, save)
        if self.plot:   self.qc.draw(output='mpl', filename='..\FYP Logbook\Diagrams\QRCM_circuit.png')