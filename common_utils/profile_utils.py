import time
import os
import sys
import psutil
import cProfile
import pstats
import io
from functools import wraps
from enum import Enum



class MemoryUnits(Enum):
    Bs = 1
    KBs = 1e3
    MBs = 1e6
    GBs = 1e9


try:
    import resource
    
    def point_statistic(point=""):
        ''' Unix specific statistic function
        '''
        usage=resource.getrusage(resource.RUSAGE_SELF)
        return "{}: usertime={} systime={} mem={} mb".format(point,usage[0],usage[1],(usage[2]*resource.getpagesize())/1e6 )
except ImportError:

    def point_statistic(point=""):
        return "{}: point_statistic works for *nix-based sytems only".format(point)


def timer_sec_wrapper(function):
    '''Time usage wrapper for heavy computations functions

    Usage:

    @timer_sec_wrapper
    def function(args):
        ...

    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running {}: {} seconds".format(function.__name__, t1-t0))
        return result
    return function_timer


def simple_profile(units=MemoryUnits.Bs):
    '''Simple time and memory usage profiling wrapper

    Usage:

    @simple_profile()
    def function(args):
        ...

    @simple_profile(units=MemoryUnits.Bs)
    def function(args):
        ...

    '''
    assert isinstance(units,MemoryUnits)
    def get_process_memory(divider):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss/divider

    def decorator(function): 
        @wraps(function)
        def wrapper(*args, **kwargs):
            mem_before = get_process_memory(units.value)
            start = time.time()
            result = function(*args, **kwargs)
            elapsed_time = time.time() - start
            mem_after = get_process_memory(units.value)
            print("{0}: memory before: {1} {5}, after: {2} {5}, consumed: {3} {5}; exec time: {4} sec".format(
                function.__name__,mem_before, mem_after, mem_after - mem_before,
                elapsed_time, units.name))
            return result
        return wrapper
    return decorator




def sizeof(obj,units=MemoryUnits.Bs):
    ''' Returns size of given object or -1
    '''
    return sys.getsizeof(obj,default=-1)/units.value


def full_cprofiler(print_statistic=True,dump_statistic=True,fileprefix=""):
    '''Function workflow statistic collecting wrapper

    Arguments:
        print_statistic : (bool) if True will print collected statistic sorted in cumulative way
        dump_statistic  : (bool) if True statistic will be dumped to file
        fileprefix      : (str) prefix of statistic file dump 

    Usage:

    @full_cprofiler()
    def function(args):
        ...

    @full_cprofiler(print_statistic=True,dump_statistic=True,fileprefix="")
    def function(args):
        ...
    '''
    def decorator(function):
        @wraps(function)
        def function_timer(*args, **kwargs):
            profiler = cProfile.Profile()
            result = profiler.runcall(function,*args, **kwargs)
            if print_statistic:
                profiler.print_stats('cumulative')
            if dump_statistic:
                profile_filename = fileprefix +"profile_" + function.__name__ + '.prof'
                profiler.dump_stats(profile_filename)
            return result
        return function_timer
    return decorator



def readStatisticFromFile(filename):
    '''Reader for .prof dumped by @full_cprofiler
    '''
    s = io.StringIO()
    stat = pstats.Stats(filename,stream=s)
    stat.sort_stats('cumulative').print_stats()
    return s.getvalue()