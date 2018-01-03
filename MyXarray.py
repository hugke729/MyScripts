import numpy as np
from xarray import DataArray


def xa_gradient(data_array, dim, dx=None):
    """Apply np.gradient in xarray fashion and return xarray.DataArray object

    Inputs
    ------
    data_array: xarray.DataArray object
        DataArray containing object to be applied
    dim: str
        Dimension on which to apply gradient. E.g., 'X', 'Y', 'T'
    dx: array
        Must be of same length as corresponding dimension of data_array

    Returns
    -------
    new_array: xarray.DataArray object
        Contains all the same attributes and coords as input data_array
    """
    coord_lengths = [data_array[coord].size for coord in data_array.coords]
    if (0 in coord_lengths) or (1 in coord_lengths):
        print('xa_gradient may be wrong if coordinates with only one value'
              'exist. For example, time slices')

    # Determine integer index associated with dim (throw warning above
    # for special case)
    axis = [i for i, coord in enumerate(data_array.coords) if coord == dim][0]
    grad_out = np.gradient(data_array, axis=axis)

    grad_out = DataArray(
        grad_out, data_array.coords, data_array.dims, data_array.attrs)

    if dx is not None:
        # Make dx a DataArray to avoid having to use Numpy's broadcasting
        dx = DataArray(dx, data_array[dim].coords, dim)
        grad_out /= dx

    return grad_out


def xa_masked_equal(data_array, value):
    """Like np.ma.masked_equal, but for xarray datasets"""
    is_equal = data_array != value
    data_array = data_array.where(is_equal)
    return data_array
