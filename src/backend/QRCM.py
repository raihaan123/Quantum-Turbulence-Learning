from qiskit import QuantumCircuit, QuantumRegister, execute, Aer
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from tqdm import tqdm

# Local imports
from tools.decorators import debug, hyperparameters


class QRCM:
    """
    Class for the QRCM - Quantum Reservoir Computing Model

    1) Intialize the qubits to |0>, the input data is given as |x^t>.
    2) Apply unitary matrices U(beta), U(4πx^t), U(4πp^t) to the qubits to evolve the state of the reservoir |ψ^(t+1)>.
    3) Measure the final state of the reservoir to get the probability vector P^(t+1).
    4) Calculate the state of the dynamical system X^(t+1) from P^(t+1) using the output weight matrix W_out.
    5) Repeat steps 2-4 for T training instances with data instances {x^(t+1), x_tg^(t+1)} where tg is target.
    6) Optimize the output weight matrix W_out using the mean squared cost function with a Tikhonov regularization term.
    7) The optimized output matrix is given by W_out* = U_tg * R^T (RR^T + beta*I)^-1, where U_tg is a matrix of target outputs and R is a matrix of reservoir states.
    8) Once the output weights are optimized and the hyperparameters are tuned, the RCM can run either in the prediction (closed-loop scenario) or reconstruction mode (open-loop scenario).


    Methods:
        - __init__(): Initialize the QRCM.
        - open_loop(): Run the QRCM in open-loop mode.
        - train(): Train the QRCM.
    """


    def __init__(self, dim=3,
                       qubits=2,
                       eps=1e-2,
                       tik=1e-2,
                       seed=0,
                       plot=False):
        """
        Initialize the QRCM.
        
        Here's a brief overview of the dimensions of the matrices and vectors involved in the QRCM:
        -	Input signal X^t: This is a vector of dimension N_in, representing the input signal to the system at time t
        -	Quantum state |psi^t>: This is a complex vector of dimension 2^n=N_dof, representing the state of the quantum system at time t
        -	Output weight matrix W_out: This is a matrix of dimension N_output x N_dof, where N_output represents the number of outputs desired from the system. The matrix maps the information stored in the quantum state to the desired output, effectively reducing the high-dimensional information in the quantum state to a lower-dimensional representation that can be used for further analysis or control
        -	Evolved probability vector P_tilde^(t+1): This is a vector of dimension N_dof, representing the probability vector (squared amplitudes) after temporal evolution through the reservoir
        -	Final probability vector P^(t+1): This is a vector of dimension N_dof, representing the final probability vector after the linear combination of the evolved probability vector and the previous probability vector has been performed using the leaking rate epsilon
        """

        ### Defining attributes of the QRCM ###

        self.N_in           = dim                               # Number of degrees of freedom of dynamical system
        # n = self.N_qubits   = int(np.ceil(np.log2(dim)))      # Minimum number of qubits required - can be increased for better performance
        n = self.N_qubits   = qubits                            # ...Override!
        N   = self.N_dof    = 2**n                              # Number of degrees of freedom of quantum reservoir
        self.qr             = QuantumRegister(n)                # Define the quantum registers in the circuit
        self.qc             = QuantumCircuit(self.qr)           # Note that the qubits are initialized to |0> by default

        self.seed           = seed                              # Set the seed for the random number generator
        self.rnd            = np.random.RandomState(seed)       # Random state object

        self.psi            = np.zeros((N))                     # |psi^t> is the quantum state - this is a complex vector of dimension 2^n
        self.X              = np.zeros((dim))                   # X^t is the latest input signal - this is a vector of dimension N_in
        self.P              = self.rnd.dirichlet(np.ones(N))    # P^t is the probability amplitude vector - this is a real vector of dimension N_dof
        self.beta           = self.rnd.uniform(0, 2*pi, n)      # Beta is random rotation vector - this is a real vector of dimension n

        self.eps            = eps                               # Leaking rate epsilon -> P^(t+1) = epsilon*P_tilde^(t+1) + (1-epsilon)*P^(t)
        self.tikhonov       = tik                               # Tikhonov regularization parameter
        self.plot           = plot                              # Plot the circuit if True
        self.time           = None                              # Time taken to complete the simulation - set by the decorator


    def add_U(self, theta):
        """ Applies a block U(theta) to the quantum circuit
        
        Note from the paper regarding classical data loading:
            "The combination of RY and CNOT gates is continued until the last qubit is reached.
            There, the CNOT is applied to the previous qubit and if not yet finished, the constructor starts at the upper qubit again."

        Args:
            theta (float): Rotation angle vector - size N_qubits (therefore only N_qubits-1 CNOT gates are needed)

        Saves:
            self.qc (QuantumCircuit): Quantum circuit object
        """
        
        n = self.N_qubits

        # Repeat for each qubit
        for i, angle in enumerate(theta):

            j = i % n           # j is the qubit index and loops from 0 to n-1
            loop = i // n       # loop is the number of times all qubits have been used

            # Apply the RY gate
            self.qc.ry(angle, j)
            
            # If not on the last qubit, apply the CNOT gate
            if j < n-1: self.qc.cx(j, j+1)
            else:       self.qc.cx(j, j-1)    # As per the note above

                
            # Add a barrier if the last qubit has been used (just for visual clarity)
            if j == n-1:  self.qc.barrier()


    def build_circuit(self):
        """
        Build the quantum circuit!

        Args:
            None

        Saves:
            self.qc (QuantumCircuit): Quantum circuit object

        """
        
        # Delete the previous circuit - ie remove all gates
        self.qc.data = []

        # Loading the reservoir state parameters
        P = self.P
        X = self.X
        b = self.beta

        # Add the unitary transformations to the circuit separated by barriers
        # The first unitary is U(4pi*P^t) followed by U(4pi*X^t) and finally U(beta)
        self.add_U(4 * pi * P)
        self.qc.barrier()
        self.add_U(4 * pi * X)
        self.qc.barrier()
        self.add_U(b)
        
        # Plot the circuit using the built-in plot function
        if self.plot:   self.qc.draw(output='mpl', filename='..\Quantum Turbulence Learning\Diagrams\QRCM_circuit.png')


    def open_loop(self):
        """ Run the QRCM in open-loop mode
        
        Args:
            shots (int): Number of shots to run the circuit for
            
        Returns:
            self.psi (np.ndarray): Evolved probability vector
            self.P_tilde (np.ndarray): Final probability vector
            self.P (np.ndarray): Updated probability vector
            self.Y (np.ndarray): Output signal
        """

        # Build the circuit
        self.build_circuit()

        # Run the circuit - save probability vector using statevector_simulator
        self.psi = np.abs(execute(self.qc, Aer.get_backend('statevector_simulator')).result().get_statevector())
        self.P_tilde = np.abs(self.psi)**2

        self.P = self.eps * self.P_tilde + (1 - self.eps) * self.P                  # Solve for the final probability vector P^(t+1)
        assert np.isclose(np.sum(self.P), 1), "Probability vector is not valid!"    # Assert that the probability vector is valid


    @hyperparameters
    def train(self, data):
        """ Train the QRCM
        
        Args:
            data (dict): Dictionary containing the training data
            
        Saves:
            self.W_out (np.ndarray): Output weight matrix (optimized via ridge regression)
        """

        # Save data to attributes
        U_washout = self.U_washout = data['U_washout']
        U_train   = self.U_train   = data['U_train']
        Y_train   = self.Y_train   = data['Y_train']
        U_test    = self.U_test    = data['U_test']
        Y_test    = self.Y_test    = data['Y_test']
        
        # Set samp to number of rows in U_train
        self.samp = len(U_train)
        self.ntest = len(U_test)

        # Cheeky way to test Wout
        # U_test = U_train
        # Y_test = Y_train

        # For each row in U_washout, build the circuit and run it - no need to save the output Y
        with tqdm(total=len(U_washout), desc="Washout") as pbar:
            for i, row in enumerate(U_washout):
                pbar.update(1)
                # print(f"\nRow {i+1}/{len(U_washout)} : X = {row}")
                self.X = row
                self.open_loop()
        
        # Plot the latest circuit
        # self.qc.draw(output='mpl', filename='..\Quantum Turbulence Learning\Diagrams\QRCM_circuit.png')

        # For each row in U_train, build the circuit and run it - save the reservoir state psi^t in R
        # W_out* = U_tg * R^T (RR^T + beta*I)^-1
        # In AX=B form, A = R^T, B = U_tg, X = W_out
        # W_out has dimensions [N_in x N_dof], U_tg has dimensions [N_in x N_train], and R has dimensions [N_dof x N_train]
        # U_tg is the target output matrix, and is the same as Y_train^T - dimensions [N_train x N_in]' = [N_in x N_train]

        # Target output matrix
        self.U_tg = Y_train.T
        self.R = np.zeros((self.N_dof, len(U_train)))

        # Add a status bar to the termainal - tqdm is a progress bar library
        with tqdm(total=len(U_train), desc="Training") as pbar:
            for i, row in enumerate(U_train):
                pbar.update(1)
                # print(f"\nRow {i+1}/{len(U_train)} : X = {row}")

                self.X = row
                self.open_loop()
                self.R[:,i] = self.psi        # Append the output signal to the R matrix

        # Calculate the optimal output weight matrix
        self.W_out = np.dot(self.U_tg, np.dot(self.R.T, np.linalg.inv(np.dot(self.R, self.R.T) + self.tikhonov * np.eye(self.N_dof))))
        
        # Save the output weight matrix to a file - try deserializing it
        np.save("log/W_out.npy", self.W_out)        # To load to self.W_out, use np.load("W_out.npy") inside the __init__ method

        Y_pred = np.zeros_like(Y_test)

        with tqdm(total=len(U_test), desc="Testing") as pbar:
            for i, row in enumerate(U_test):
                pbar.update(1)
                # print(f"\nRow {i+1}/{len(U_test)} : X = {row}")
                self.X = row
                self.open_loop()
                Y_pred[i] = np.dot(self.W_out, self.psi)         # Multiply psi by W_out to get the output signal

        # Calculate the absolute error for each timestep - for each N_in on a new line plot
        self.err_ts = np.abs(Y_test - Y_pred)

        # Plot the error time series for all N_in dimensions
        # plt.figure()
        
        # for i in range(self.N_in):
        #     plt.plot(self.err_ts[:,i], label=f"Dimension {i+1}")
        # plt.title("Absolute Error Time Series")
        # plt.legend()
        # plt.xlabel("Time Step")
        # plt.ylabel("Absolute Error")
        # plt.ylim(0, 1.1 * np.max(self.err_ts))      # make sure the data covers 70% of the plot height
        # plt.show()
        
        # Find MSE
        self.MSE = np.mean(self.err_ts**2)
        self.MSE_full = np.mean(self.err_ts**2, axis=0)