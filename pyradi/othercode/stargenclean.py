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
# from scipy import ndimage
from scipy import misc

# ---- Implementation ----------------------------------------------- #

def draw_siemens_star(origin, radius, n, outfile,dpi):
    # Create figure and add patterns
    fig, ax = plt.subplots()
    ax.add_collection(gen_siemens_star(origin, radius, n))
    plt.axis('equal')
    plt.axis([-1.03, 1.03, -1.03, 1.03])
    plt.axis('off')
    fig.savefig(outfile, figsize=(900,900), papertype='a0', bbox_inches='tight', dpi=dpi)
    #read image back in order to crop to spokes only
    imgIn = np.abs(255 - misc.imread(outfile)[:,:,0])
    nz0 = np.nonzero(np.sum(imgIn,axis=0))
    nz1 = np.nonzero(np.sum(imgIn,axis=1))
    imgOut = imgIn[(nz1[0][0]-1) : (nz1[0][-1]+2),  (nz0[0][0]-1) : (nz0[0][-1]+2)]
    imgOut = np.abs(255 - imgOut)
    misc.imsave(outfile, imgOut)


def gen_siemens_star(origin, radius, n):
    centres = np.linspace(0, 360, n+1)[:-1]
    step = (((360.0)/n)/4.0)
    patches = []
    for c in centres:
        patches.append(Wedge(origin, radius, c-step, c+step))
    return PatchCollection(patches, facecolors='k', edgecolors='none')


def main():
    opt = docopt(__doc__)
    opt_n = int(opt['-n'])
    opt_debug = opt['--debug']
    opt_dpi = int(opt['--dpi'])
    opt_show = opt['--show']
    opt_output = opt['<output>']
    draw_siemens_star((0,0), 1., opt_n, opt_output, opt_dpi)


if __name__ == '__main__':
    main()
