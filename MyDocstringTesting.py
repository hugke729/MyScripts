# Sample file for making neatly formatted docstrings
# Based on Sphinx/numpydoc
# When using Spyder, I notice that a host of rich text options are available
# and I want to learn to do this myself

# Testing by creating dummy functions and then checking what their docstrings
# look like in Spyder's object inspector

# See also
# http://matplotlib.org/sampledoc/cheatsheet.html

# Much of it appears similar to the Markdown that I'm familiar with


def basic_formatting():
    """
    *italics*, **bold**, ``monotype``
    """
    x = 1
    return x


def code_block():
    """
    Some code::

        import numpy as np
        x = np.r_[:5]

    Note (1) blank line between description and code and (2) double colon.
    """


def headings():
    """
    ************
    Main heading
    ************

    Subheading
    ==========

    Smaller Subheading
    ------------------
    """


def lists():
    """
    Bulleted list
    -------------

    * Point A

    * Point B

    Enumerated list
    ---------------

    #. Point one

    #. Point two
    """


def parameter_lists():
    """
    Returns
    -------
    out1 :
        value without type
    out2 : dict
        value with type
    out3 : no second line
    """


def linebreaks():
    """
    |  First line
    |  Second line
    |  Notice double space
    """


def emphasise():
    """
    .. code-block:: python
       :emphasize-lines: 3,5

       def some_function():
           interesting = False
           print 'This line is highlighted.'
           print 'This one is not...'
           print '...but this one is.'
    """


def alternative_parameter_method(one, two):
    """
    Inputs
    ------

    :param one: describe one
    :param two: describe two
    """
