##collapse into single dimension
#imgFlat = img.flatten()
#
#imgFlatSort = numpy.sort(imgFlat)
#
##imgFlatSort = (imgFlat)
##cumulative distribution
#cdf = imgFlatSort.cumsum()/imgFlatSort[-1]
#y=numpy.interp(imgFlat,imgFlatSort, cdf )
#imgHEQ = y.reshape(img.shape)

#uncomment the next few lines to force same colors for all images
#for entry in framesToLoad:
#    minAllImages = numpy.min(y)
#    maxAllImages = numpy.max(y)
#    # we force image(0,0) to minium over all images
#    imgHEQ[entry-1][0][0] = minAllImages
#    # we force image(0,0) to max over all images
#    imgHEQ[entry-1][numrows-1][numcols-1] = maxAllImages


#minData = numpy.min(imgFlat)
#maxData = numpy.max(imgFlat)
#print('Image irradiance range minimum={0} maximum={1}'.format(minData, maxData))
#irradRange=numpy.linspace(minData, maxData, 100)
#print(irradRange.shape)
#normalRange = numpy.interp(irradRange,imgFlatSort, cdf )
#print(normalRange.shape)
#H = ryplot.Plotter(1, 1, 1,'Mapping Input Irradiance to Equalised Value', figsize=(10, 10))
#H.plot(1, "","Irradiance [W/(m$^2$)]", "Equalised value",irradRange , normalRange, powerLimits = [-4,  2,  -10,  2])
#H.getPlot().show()
#H.saveFig('cumhist.png', dpi=300)
#
##now we attempt to change the indices on the color map, to reflect the original input irradiance values
##see http://www.scipy.org/Cookbook/Matplotlib/ColormapTransformations
## start with jet and then change the indices

cmap=plt.cm.jet
#cdict = cmap._segmentdata
#
#irradRange[0]=0
#normalRange[0]=0
#normalRange = normalRange/numpy.max(normalRange)
#irradRange = irradRange/numpy.max(irradRange)
#print(min(normalRange),  max(normalRange))
#print(min(irradRange),  max(irradRange))
#function = interpolate.interp1d(normalRange,  irradRange)
#
#print(0.5, function(0.5)[()],  type(function(0.5)[()]))
#function_to_map = lambda x : ((function(x[0]))[()], x[1], x[2])
#
#print(map(function_to_map, ( (0, .4, .2),(0.3, .5, .3)  ) ))
#
#print(cdict['red'])
#for key in ('red', 'green', 'blue'):
#    cdict[key] = [function_to_map(var) for var in cdict[key] ]  #            map(function_to_map, cdict[key] )
#    cdict[key] .sort()
#    cdict[key] = tuple(cdict[key])
#    assert (cdict[key][0]<0 or cdict[key][-1]>1), "Resulting indices extend out of the [0, 1] segment."
#
#print(cdict['red'])
#
#print(type(cdict['red'][0][0]))
#print(cdict['red'][0][0])
