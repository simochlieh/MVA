import getopt
from math import *
import matplotlib.pyplot as plt
import numpy as np
import parser
import sys


def plot_f(expression, x_from, x_to, interval):
    code = parser.expr(expression).compile()
    x_range = np.arange(x_from, x_to, interval)
    y = [eval(code) for x in x_range]
    plt.plot(x_range, y)
    plt.show()


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "he:f:t:i:", ["help", "expression=", "from=", "to=", "interval="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err)  # will print something like "option -a not recognized"
        sys.exit(2)
    x_from = 0.
    x_to = 10.
    interval = 1.
    expression = "x**2"
    for o, a in opts:
        if o in ("-e", "--e"):
            expression = a
        elif o in ("-f", "--from"):
            x_from = float(a)
        elif o in ("-t", "--to"):
            x_to = float(a)
        elif o in ("-i", "--interval"):
            interval = float(a)
        elif o in ("-h", "--help"):
            print("""
                Usage: python plot_f.py [options]
            
                General Options:
                -h, --help                  Show help.
                -y, --function              Expression to plot, the variable must be named 'x'
                -f, --from                  lower bound on x-axis 
                -t, --to                    upper bound on x-axis
            """)
            sys.exit()
        else:
            assert False, "unhandled option"
    plot_f(expression, x_from, x_to, interval)

if __name__ == "__main__":
    main()


