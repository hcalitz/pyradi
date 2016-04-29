#!/usr/bin/env python

"""Siemens star chart generator
http://cmp.felk.cvut.cz/~wagnelib/utils/star.html

by Libor Wagner

Usage:
    stargen.py [options] <output>

Options:
    -h, --help              Show this message
    -n N                    Number of rays [default: 100]
    --show                  Show the preview of the start
    --dpi DPI               DPI of the generated image [default: 600]
    --debug                 Run in debug mode
"""

# ---- Imports ------------------------------------------------------ #

from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection

# ---- Implementation ----------------------------------------------- #


def main():
    opt = docopt(__doc__)
    opt_n = int(opt['-n'])
    opt_debug = opt['--debug']
    opt_dpi = int(opt['--dpi'])
    opt_show = opt['--show']
    opt_output = opt['<output>']

    # Create figure
    fig, ax = plt.subplots()

    # Generate patters add text
    ax.add_collection(gen_siemens_star((0, 0), 1, opt_n))
    # ax.text(1.05, 0.90, 'N=%d' % opt_n, fontsize=15)

    # Plot
    plt.axis('equal')
    plt.axis([-1.03, 1.03, -1.03, 1.03])
    plt.axis('off')
    # Safe figure
    fig.savefig(opt_output, figsize=(8000, 6000), papertype='a0', bbox_inches='tight', dpi=opt_dpi)

    # Show if required
    if opt_show:
        plt.show()


def gen_siemens_star(origin, radius, n):
    centres = np.linspace(0, 360, n+1)[:-1]
    step = (((360.0)/n)/4.0)
    patches = []
    for c in centres:
        patches.append(Wedge(origin, radius, c-step, c+step))
    return PatchCollection(patches, facecolors='k', edgecolors='none')


if __name__ == '__main__':
    main()
