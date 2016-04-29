"""This script opens and plots OSSIM hires raw double irradiance files.

The problem is that the dynamic range in most images is very, very wide,
hiding some details in the low irradiance levels.

This script plots the images after considerable non-linear compression,
BUT the colour map tick labels are true to the original input values, so
you can read the true image irradiance values from the labels.

The data is read in, compressed by some math function and then histogram equalised.
The tick labels on the color bar is  calculated from the inverse function by which the image is compressed.

Please let me know if you improve this script.
neliswillers@gmail.com

There is some really nice python tricks in this script - this can most definately not be done in matlab!
"""

import math
import numpy
import pyradi.ryfiles as ryfiles
import pyradi.ryplot as ryplot
import matplotlib.pyplot as plt
from scipy import interpolate

#requires pyradi r70 or later

#image size
numrows = 512
numcols = 512
# select the frames to load
framesToLoad = [0, 9, 19]
imagefile1='tp13a_AS30Missile_C1.double'
imagefile3='tp13a_AS30Missile_C3.double'

#below we compress the image (and then inversely expand the color bar values), prior to histogram equalisation
#to ensure that the two keep in step, we store the function names as pairs, and below invoke the pair elements
# cases are as follows:  linear, log. sqrt.  Note that the image is histogram equalised in all cases.
compressSet = [
               [lambda x : x , lambda x : x, 'Linear'],
               [numpy.log,  numpy.exp, 'Natural Log'],
               [numpy.sqrt,  numpy.square, 'Square Root']]
#select of the three above
selectCompressSet = 2



## second case single colour

frames, img = ryfiles.readRawFrames(imagefile1, numrows, numcols, numpy.float64, framesToLoad)

if  not frames == len(framesToLoad):
    print('Data not loaded: single colour plot')
else:
    plotTitle = 'Image Irradiance ({0} then Equalised) [W/m$^2$]'.format(compressSet[selectCompressSet][2])
    P = ryplot.Plotter(1, 2, 2,plotTitle, figsize=(15, 15))
    #prepare the color bar and plot first frame on linear scale
    imgLevels = numpy.linspace(numpy.min(img[0]), numpy.max(img[0]), 20)
    customticksz = zip(imgLevels, ['{0:10.3e}'.format(x) for x in imgLevels])
    P.showImage(1, img[0], 'frame {0}'.format(framesToLoad[0]), cbarshow=True, \
                        cbarcustomticks=customticksz, cbarfontsize=5)

    #now do for more frames, but compression
    for entry in framesToLoad:
        #first use histogram equalization to remap the pixel values
        #collapse into single dimension
        #NB!! compress the input image by taking square root of irradiance - rescale color bar tick below to match!!!!!!
        imgFlat = compressSet[selectCompressSet][0](img[framesToLoad.index(entry)].flatten())
        imgFlatSort = numpy.sort(imgFlat)
        #cumulative distribution
        cdf = imgFlatSort.cumsum()/imgFlatSort[-1]
        #remap image values to achieve histo equalisation
        y=numpy.interp(imgFlat,imgFlatSort, cdf )
        #and reshape to image shape
        imgHEQ = y.reshape( img[framesToLoad.index(entry)].shape)

#        #plot the histogram mapping
#        minData = numpy.min(imgFlat)
#        maxData = numpy.max(imgFlat)
#        print('Image irradiance range minimum={0} maximum={1}'.format(minData, maxData))
#        irradRange=numpy.linspace(minData, maxData, 100)
#        normalRange = numpy.interp(irradRange,imgFlatSort, cdf )
#        H = ryplot.Plotter(1, 1, 1,'Mapping Input Irradiance to Equalised Value', figsize=(10, 10))
#        H.plot(1, "","Irradiance [W/(m$^2$)]", "Equalised value",irradRange , normalRange, powerLimits = [-4,  2,  -10,  2])
#        #H.getPlot().show()
#        H.saveFig('cumhist{0}.png'.format(entry), dpi=300)

        #prepare the color bar tick labels from image values (as plotted)
        imgLevels = numpy.linspace(numpy.min(imgHEQ), numpy.max(imgHEQ), 20)
        #map back from image values to original values as read it (inverse to above)
        irrLevels=numpy.interp(imgLevels,cdf, imgFlatSort)
        #NB!! uncompress the lick labels by taking square of image values - match this with compression above !!!!!
        customticksz = zip(imgLevels, ['{0:10.3e}'.format(compressSet[selectCompressSet][1](x)) for x in irrLevels])
        P.showImage(framesToLoad.index(entry)+2, imgHEQ,  'frame {0}'.format(entry),\
                          cmap=plt.cm.jet , cbarshow=True, cbarcustomticks=customticksz, cbarfontsize=5)

    P.getPlot().show()
    P.saveFig('imageCompressEqualise.png', dpi=300)
    print('\n{0} frames of size {1} x {2} and data type {3} read from binary file {4}'.\
          format(img.shape[0],img.shape[1],img.shape[2],img.dtype, imagefile))



## second case multiple colours
framesToLoad = [0, 19]
frames1, img1 = ryfiles.readRawFrames(imagefile1, numrows, numcols, numpy.float64, framesToLoad)
frames3, img3 = ryfiles.readRawFrames(imagefile3, numrows, numcols, numpy.float64, framesToLoad)
#concatenate the two colours as one long row
flat = numpy.concatenate([ img1.flatten(),  img3.flatten()])
#now reshape with colour as first index
img = flat.reshape(2, len(framesToLoad), numrows, numcols)
specband = ['3-5 $\mu$m','1.5-2.5 $\mu$m'  ]
if not (frames1 == len(framesToLoad) and  frames3 == len(framesToLoad)):
    print('Data not loaded: two-colour plot')
else:
    #now do for more frames, but compression
    for entry in framesToLoad:
        print(entry)
        plotTitle = 'Image Irradiance ({0} then Equalised) [W/m$^2$]'.format(compressSet[selectCompressSet][2])
        P = ryplot.Plotter(1, 1, 2,plotTitle, figsize=(15, 15))
        for colourIndex in [0, 1]:
            #first use histogram equalization to remap the pixel values
            #collapse into single dimension
            #NB!! compress the input image by taking square root of irradiance - rescale color bar tick below to match!!!!!!
            imgFlat = compressSet[selectCompressSet][0](img[colourIndex][framesToLoad.index(entry)].flatten())
            imgFlatSort = numpy.sort(imgFlat)
            #cumulative distribution
            cdf = imgFlatSort.cumsum()/imgFlatSort[-1]
            #remap image values to achieve histo equalisation
            y=numpy.interp(imgFlat,imgFlatSort, cdf )
            #and reshape to image shape
            imgHEQ = y.reshape( img[colourIndex][framesToLoad.index(entry)].shape)

            #prepare the color bar tick labels from image values (as plotted)
            imgLevels = numpy.linspace(numpy.min(imgHEQ), numpy.max(imgHEQ), 20)
            #map back from image values to original values as read it (inverse to above)
            irrLevels=numpy.interp(imgLevels,cdf, imgFlatSort)
            #NB!! uncompress the lick labels by taking square of image values - match this with compression above !!!!!
            customticksz = zip(imgLevels, ['{0:10.3e}'.format(compressSet[selectCompressSet][1](x)) for x in irrLevels])
            #P.showImage(colourIndex+1, imgHEQ,  '{0}, Frame {1}'.format(specband[colourIndex], entry),\
            #                  cmap=plt.cm.jet , cbarshow=True, cbarcustomticks=customticksz, cbarfontsize=10)
            P.showImage(colourIndex+1, imgHEQ,  '{0}, Frame {1}'.format(specband[colourIndex], entry),\
                              cmap=plt.cm.jet )

        #P.getPlot().show()
        P.saveFig('imageCompressEqualiseC1C3{0}.png'.format(entry), dpi=300)

exit()


#http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps
#http://www.scipy.org/Cookbook/Matplotlib/ColormapTransformations
#http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
#http://stackoverflow.com/questions/9141732/how-does-numpy-histogram-work
#http://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
#http://matplotlib.sourceforge.net/examples/pylab_examples/show_colormaps.html


