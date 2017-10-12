
import numpy as np


def count_lines(filepath):
    return sum(1 for _ in open(filepath))


def parse_data_with_labels(filepath, dimension, delimiter):
    """
    Parses the data stored in filepath into 2 numpy arrays
    data_x (nb_lines*dimension) and data_y (nb_lines*1) with float values.
    The file shouldn't have any header and values are supposed to be numeric

    :param filepath: string path of the data file
    :param dimension: int
    :param delimiter: string
    :return: data_x, data_y
    """
    nb_lines = count_lines(filepath)
    data_x = np.empty(shape=[nb_lines, dimension])
    data_y = np.empty(shape=[nb_lines, 1])
    try:
        with open(filepath) as f:
            i = 0
            for line in f:
                data_x[i] = map(float, line.strip().split(delimiter))[:-1]
                data_y[i] = map(float, line.strip().split(delimiter))[-1:]
                i += 1
    except IOError:
        print "file %s not found" % filepath

    return data_x, data_y
