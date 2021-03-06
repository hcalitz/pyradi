import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
import os
import pkg_resources
from StringIO import StringIO
import pyradi.ryfiles as ryfiles
from numbers import Number


##############################################################################################
##############################################################################################
##############################################################################################
class Spectral(object):
    """Generic spectral can be used for any spectral vector
    """
    ############################################################
    ##
    def __init__(self, ID, value, wl=None, wn=None, desc=None):
        """Defines a spectral variable of property vs wavelength or wavenumber

        One of wavelength or wavenunber must be supplied, the other is calculated.
        No assumption is made of the sampling interval on eiher wn or wl.

        The constructor defines the 

            Args:
                | ID (str): identification string
                | wl (np.array (N,) or (N,1)): vector of wavelength values
                | wn (np.array (N,) or (N,1)): vector of wavenumber values
                | value (np.array (N,) or (N,1)): vector of property values
                | desc (str): description string

            Returns:
                | None

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', ]

        self.ID = ID
        self.desc = desc

        self.wn = wn
        self.wl = wl
        if wn is not None:
            self.wn =  wn.reshape(-1,1)
            self.wl = 1e4 /  self.wn
        elif wl is not None:
            self.wl =  wl.reshape(-1,1)
            self.wn = 1e4 /  self.wl
        else:
            pass

        if isinstance(value, Number):
            if wn is not None:
                self.value = value * np.ones(self.wn.shape)
            else:
                self.value = value 
        elif isinstance(value, np.ndarray):
            self.value = value.reshape(-1,1)


    ############################################################
    ##
    def __str__(self):
        """Returns string representation of the object

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        if isinstance(self.wn, np.ndarray):
            numpts = self.wn.shape[0]
            stride = numpts / 3

        strn = '{}\n'.format(self.ID)
        strn += ' {}-desc: {}\n'.format(self.ID,self.desc)
         # for all numpy arrays, provide subset of values
        for var in self.__dict__:
            # then see if it is an array
            if isinstance(eval('self.{}'.format(var)), np.ndarray):
                svar = (np.vstack((eval('self.{}'.format(var))[0::stride], eval('self.{}'.format(var))[-1] ))).T
                strn += ' {}-{} (subsampled.T): {}\n'.format(self.ID,var, svar)
            elif isinstance(eval('self.{}'.format(var)), Number):
                svar = eval('self.{}'.format(var))
                strn += ' {}-{} (subsampled.T): {}\n'.format(self.ID,var, svar)

        return strn


    ############################################################
    ##
    def __mul__(self, other):
        """Returns a spectral product

        it is not intended that the function will be called directly by the user

            Args:
                | other (Spectral): the other Spectral to be used in multiplication

            Returns:
                | str

            Raises:
                | No exception is raised.
        """

        # if isinstance(other, Number):
        #     if isinstance(self.wn, np.ndarray):
        #         other = Spectral('{}'.format(other),value=other, wn=self.wn,desc='{}'.format(other))
        #     else:
        #         other = Spectral('{}'.format(other),value=other, desc='{}'.format(other))

        if isinstance(other, Spectral):
            if self.wn is not None and other.wn is not None:
                # create new spectral in wn wider than either self or other.
                wnmin = min(np.min(self.wn),np.min(other.wn))
                wnmax = max(np.max(self.wn),np.max(other.wn))
                wninc = min(np.min(np.abs(np.diff(self.wn,axis=0))),np.min(np.abs(np.diff(other.wn,axis=0))))
                wn = np.linspace(wnmin, wnmax, (wnmax-wnmin)/wninc)
                wl = 1e4 / self.wn
                if np.mean(np.diff(self.wn,axis=0)) > 0:
                    s = np.interp(wn,self.wn[:,0], self.value[:,0])
                    o = np.interp(wn,other.wn[:,0], other.value[:,0])
                else:
                    s = np.interp(wn,np.flipud(self.wn[:,0]), np.flipud(self.value[:,0]))
                    o = np.interp(wn,np.flipud(other.wn[:,0]), np.flipud(other.value[:,0]))
            elif self.wn is     None and other.wn is not None:
                o = other.value
                s = self.value
                wl = other.wl    
                wn = other.wn    

            elif self.wn is not None and other.wn is     None:
                o = other.value
                s = self.value
                wl = self.wl    
                wn = self.wn    

            else:
                o = other.value
                s = self.value
                wl = None    
                wn = None    
            rtnVal = Spectral(ID='{}*{}'.format(self.ID,other.ID), value=s * o, wl=wl, wn=wn,
                    desc='{}*{}'.format(self.desc,other.desc))
        else:
            rtnVal = Spectral(ID='{}*{}'.format(self.ID,other), value=self.value * other, wl=self.wl, 
                wn=self.wn,desc='{}*{}'.format(self.desc,other))

        return rtnVal


    ############################################################
    ##
    def __pow__(self, power):
        """Returns a spectral to some power

        it is not intended that the function will be called directly by the user

            Args:
                | power (number): spectral raised to power

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        return Spectral(ID='{}**{}'.format(self.ID,power), value=self.value ** power, 
            wl=self.wl, wn=self.wn,desc='{}**{}'.format(self.desc,power))

    ############################################################
    ##
    def plot(self, filename, heading, ytitle=''):
        """Do a simple plot of spectral variable(s)

            Args:
                | filename (str): filename for png graphic
                | heading (str): graph heading
                | ytitle (str): graph y-axis title

            Returns:
                | Nothing, writes png file to disk

            Raises:
                | No exception is raised.
        """
        import pyradi.ryplot as ryplot
        p = ryplot.Plotter(1,2,1,figsize=(8,5))

        if isinstance(self.value, np.ndarray):
            xvall = self.wl
            xvaln = self.wn
            yval = self.value
        else:
            xvall = np.array([1,10])
            xvaln = 1e4 / xvall
            yval = np.array([self.value,self.value])

        p.plot(1,xvall,yval,heading,'Wavelength $\mu$m', ytitle)
        p.plot(2,xvaln,yval,heading,'Wavenumber cm$^{-1}$', ytitle)

        p.saveFig(ryfiles.cleanFilename(filename))




################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':
    import math
    import sys
    from scipy.interpolate import interp1d
    import pyradi.ryplanck as ryplanck
    import pyradi.ryplot as ryplot
    import pyradi.ryfiles as ryfiles
    import os

    figtype = ".png"  # eps, jpg, png
    # figtype = ".eps"  # eps, jpg, png

    doAll = False

    if True:
        spectrals = {}
        atmos = {}
        sensors = {}
        targets = {}

        # test loading of spectrals
        print('\n---------------------Spectrals:')
        spectral = np.loadtxt('data/MWIRsensor.txt')
        spectrals['ID0'] = Spectral('ID1',value=.3,desc="const value")
        spectrals['ID1'] = Spectral('ID1',value=spectral[:,1],wl=spectral[:,0],desc="spec value")

        spectrals['IDp00'] = spectrals['ID0'] * spectrals['ID0']
        spectrals['IDp01'] = spectrals['ID0'] * spectrals['ID1']
        spectrals['IDp10'] = spectrals['ID1'] * spectrals['ID0']
        spectrals['IDp11'] = spectrals['ID1'] * spectrals['ID1']

        spectrals['ID0pow'] = spectrals['ID0'] ** 3
        spectrals['ID1pow'] = spectrals['ID1'] ** 3

        spectrals['ID0mul'] = spectrals['ID0'] * 1.67
        spectrals['ID1mul'] = spectrals['ID1'] * 1.67

        for key in spectrals:
            print(spectrals[key])
        for key in spectrals:
            filename ='{}-{}'.format(key,spectrals[key].desc)
            spectrals[key].plot(filename=filename,heading=spectrals[key].desc,ytitle='Value')

