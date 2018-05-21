import sys
import re
import pickle
import os
import numpy as np
import numpy.ma as ma
from IPython import get_ipython
from jupyter_client.blocking import BlockingKernelClient


def filedrive():
    if sys.platform == 'linux':
        filedrive = '/'
    else:
        filedrive = 'F:/'
    return filedrive


def phd_folder():
    """Path to my PhD folder including .../PhD/"""
    if sys.platform == 'linux':
        phd_folder = '/home/hugke729/PhD/'
    else:
        phd_folder = 'C:/Users/Ken/Documents/PhD/'
    return phd_folder


def replace(file, pattern, subst):
    # Read contents from file as a single string
    file_handle = open(file, 'r')
    file_string = file_handle.read()
    file_handle.close()

    # Use RE package to allow for replacement
    # (also allowing for (multiline) REGEX)
    file_string = (re.sub(pattern, subst, file_string))

    # Write contents to file.
    # Using mode 'w' truncates the file.
    file_handle = open(file, 'w')
    file_handle.write(file_string)
    file_handle.close()


def rm_var_except(except_list, ns=locals()):
    """remove all variables except those given in except_list"""

    # use Ipython's who_ls to create a list containing only the variables that
    # I actually interact with
    mgc = get_ipython().magic
    all_vars = mgc(
        'who_ls ndarray list dict int str float bool tuple int32 int64')

    for item in all_vars:
        if item in except_list:
            # Do nothing
            pass
        else:
            exec('del(' + item + ')', globals(), ns)


def unpickle2(pickle_file):
    """unpickle a file created with python 2 into python 3"""
    f = open(pickle_file, 'rb')
    D = pickle.load(f, encoding='latin1')
    f.close()
    return D


def rmvar(var_list):
    for i in var_list:
        try:
            del(i)
        except:
            pass


def preall(array_shape, copies=1, initial_value=0, mask=False):
    """simulataneously preallocate zero array(s) of identical shape

    returns a tuple of n='copies' of initial_value arrays of shape array_shape
    If copies is 1, only the array is returned (i.e. not as a tuple)
    If mask is True, convert to masked_array"""
    array_tuple = ()
    for _ in range(copies):
        empty_array = np.full(array_shape, float(initial_value))
        if mask:
            empty_array = ma.masked_invalid(empty_array)

        # addition with tuples appends the last value
        array_tuple += (empty_array, )

    return array_tuple[0] if copies == 1 else array_tuple


def merge_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def run_in_existing_kernel(cmd):
    kernel_dir = '/run/user/1000/jupyter/'
    connection_file = os.listdir(kernel_dir)[0]
    connection_file = kernel_dir + connection_file
    kc = BlockingKernelClient(connection_file=connection_file)
    kc.load_connection_file()
    kc.start_channels()

    kc.execute(cmd)
