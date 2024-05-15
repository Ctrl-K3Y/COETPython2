from time import perf_counter as timer

def printnp(np_array):
    """Prints out data about the shape of a numpy array"""
    print(np_array)
    print(f"Shape: {np_array.shape}, DimensionsL {np_array.ndim}")
    print(f"Datatype: {np_array.dtype}, Item Size: {np_array.itemsize}")


def time_f(function, args):
    """Takes ina function and its arguments, runs it, returns
    the result, but in the meantime prints out long it took"""
    start_time = timer()
    ret_val = function(args)
    total_time = (timer() - start_time)
    print(f"{function.__name__} took {total_time} ms to complete")
    return ret_val