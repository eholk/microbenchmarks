#!/usr/bin/python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def loadFile(name):
    data = np.loadtxt(name,
                      delimiter=",",
                      dtype = { 'names': ('size', 'time'),
                                'formats': ('int', 'float')}
                      )

    return data

def mkplot(name, xlabel, plot_name="plot", yscale=1, xscale=1, logy=False):
    harlan = loadFile('%s.csv' % (name))

    if logy:
        harlan = plt.loglog(harlan['size'] / xscale,
                            harlan['time'] / yscale,
                            'bo')
    else:
        harlan = plt.semilogx(harlan['size'] / xscale,
                              harlan['time'] / yscale,
                              'bo')

    plt.ylabel("Execution Time (ms)")
    plt.xlabel(xlabel)

    plt.tight_layout()
    plt.savefig(plot_name + '.pdf')

def do_plots():
    id = 1    

    matplotlib.rc('font', size=10)
    #matplotlib.rc('lines', linewidth=2.0)
    #matplotlib.rc('lines', markeredgewidth=2.0)
    matplotlib.rc('legend', fontsize=10)
    #size = (4, 2.5)
    size = None

    plt.figure(id, figsize=size)
    mkplot('memcopy', 'Transfer Size (bytes)', plot_name="figure5-memcopy",
           logy=True)
    id += 1
    
    plt.figure(id, figsize=size)
    mkplot('memcopy-chunked', 'Chunk Size (KB)',
           plot_name="figure6-memcopy-chunked", xscale=1024)
    id += 1

do_plots()
