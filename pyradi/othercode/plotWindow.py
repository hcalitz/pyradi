from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
import pylab

beta = [2,4,16,32]

wfun=[
numpy.bartlett, 
numpy.blackman, 
numpy.hamming, 
numpy.hanning]

pylab.figure()
#for b in beta:
# w = numpy.kaiser(11,b) 
# pylab.plot(range(len(w)),w,label="beta = "+str(b))
for wf in wfun:
 w = wf(11) 
 pylab.plot(range(len(w)),w,label="beta = ")
pylab.xlabel('n')
pylab.ylabel('W_K')
pylab.legend()
pylab.show()

#
#def smooth(x,beta):
# """ kaiser window smoothing """
# window_len=11
# # extending the data at beginning and at the end
# # to apply the window at the borders
# s = numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
# w = numpy.kaiser(window_len,beta)
# print(s.shape,  w.shape)
# y = numpy.convolve(w/w.sum(),s,mode='valid'.encode('utf-8'))
# return y[5:len(y)-5]
# 
# 
# # random data generation
#y = numpy.random.random(100)*100 
#for i in range(100):
# y[i]=y[i]+i**((150-i)/80.0) # modifies the trend
#
## smoothing the data
#pylab.figure(1)
#pylab.plot(y,'-k',label="original signal",alpha=.3)
#for b in beta:
# yy = smooth(y,b) 
# pylab.plot(yy,label="filtered (beta = "+str(b)+")")
#pylab.legend()
#pylab.show()
