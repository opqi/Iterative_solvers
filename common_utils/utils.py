import time
from functools import wraps

 
def logging(dump_to_file=False,filename="log",verbose=True):
    """A decorator logging inputs of functions
    """
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            res = function(*args, **kwargs)
            s = "Time {:.5} : {} has been called with args:\n{}\n{}".format(time.time() - wrapper.start_time,function.__name__, args, kwargs)
            if dump_to_file:
                filename_ = filename+".lgf"
                with open(filename_,"a") as file:
                    file.write(s)
            if verbose:
                print(s)
            return res
        wrapper.start_time = time.time() 
        return wrapper
    return decorator
 
 
def counter(function):
    """A decorator counts and prints the number of times a function has been executed
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        result = function(*args, **kwargs)
        print("Time {:.2} : {} has been used: {}x times".format(time.time() - wrapper.start_time,function.__name__, wrapper.count))
        return result
    wrapper.count = 0
    wrapper.start_time = time.time() 
    return wrapper