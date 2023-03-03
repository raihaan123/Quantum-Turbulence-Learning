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

# print(debug.__doc__)