# Python Decorators

import time

def debug(func):
    """
    A debug decorator

        - Outputs the function name and arguments (name, value) passed to the function
        - Outputs the time taken to execute the function
        - Outputs the return value (name, value) returned by the function

    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print(f"\nExecuting {func.__name__}")
        print(f"Arguments: {args}, {kwargs}")
        print(f"Time taken: {end_time-start_time}")
        print(f"Return value: {result}")

        return result

    return wrapper


def hyperparameters(func):
    """
    Prints the hyperparameters of the ESN object
    - self.N_qubits
    - self.eps
    
    Also logs the hyperparameters to a file, including the results of the training
    """
    
    def wrapper(*args, **kwargs):

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print(f"\nTrain/test with {args[0].samp}/{args[0].ntest} samples")      # This is a bit hacky
        print(f"Using n = {args[0].N_qubits} qubits and epsilon = {args[0].eps}")        
        print(f"\nTraining MSE: {args[0].MSE}")
        print(f"Time taken: {end_time-start_time}")
        args[0].time = end_time-start_time
        
        # Log the hyperparameters to a file
        with open("log/log.txt", "a") as f:
            f.write(f"sam={args[0].samp}/{args[0].ntest}, n={args[0].N_qubits}, eps={args[0].eps}, mse={args[0].MSE}, t={end_time-start_time}\n")
        
        return result

    return wrapper