from PIL import Image
import numpy as np
import math
from scipy import signal

def boxfilter(n):
    assert n%2!=0 # make sure it's odd
    value = 1.0/(n*n)
    filter = np.zeros(shape=(n,n), dtype='d')
    filter.fill(value)
    assert np.round(np.sum(filter))==1.0
    return filter

def gauss1d(sigma):
    length = int(np.ceil(6*sigma))
    if (length%2==0):
        length += 1
    values = np.array([x for x in xrange((-length/2)+1,(length/2)+1)], dtype='d') # create np array with doubles
    values = np.exp(-(values**2)/(2*(sigma**2)))
    values = (1/np.sum(values))*values
    return values

def gauss2d(sigma):  
    x = gauss1d(sigma)
    x = x[np.newaxis]

    y = gauss1d(sigma)
    y = y[np.newaxis]
    y = y.transpose()

    return signal.convolve2d(x,y)

def gaussconvolve2d(image, sigma):
    i = Image.open(image) 
    i = i.convert('L')

    i = np.asarray(i) 
    f = gauss2d(sigma)
    i_filtered = signal.convolve2d(i, f, 'same')
    
    new_image = Image.fromarray(i_filtered.astype('uint8'), 'L')
    new_image.save('/home/i/i7f7/cs425/a1/new.png','PNG')

### TESTS ###

def testBoxFilter():
    # check that our boxfilter works correctly for n=3,4,5...
    for n in [3,4,5]:
        try:
            print 'Boxfilter with n =', n 
            print boxfilter(n)
        except AssertionError:
            # catch where it doesn't work
            print 'Couldnt generate boxfilter of size ',n,'x',n

def testGauss1D():
    for n in [0.3,0.5,1.0,2.0]:
        try : 
            val = gauss1d(n)
            assert np.round(np.sum(val))==1 # check that it is normalized
            print '1D Gauss filter with sigma = ',n, 'length', val.size, np.sum(val), '\n', val
        except AssertionError:
            print 'Did not correctly generate a 1D Gaussian filter!', np.sum(val), val.size

def testGauss2D():
    for n in [0.5,1.0]:
        try : 
            val = gauss2d(n)
            assert np.round(np.sum(val))==1 # check that it is normalized
            print '2D Gauss filter with sigma = ',n, 'length', val.size, np.sum(val), '\n', val
        except AssertionError:
            print 'Did not correctly generate a 2D Gaussian filter!', np.sum(val), val.size


def testGaussFilter():
    gaussconvolve2d('/home/i/i7f7/cs425/a1/peacock.png',3)

### MAIN ###
if __name__=="__main__":
    gauss1d(.75)
    testBoxFilter() 
    testGauss1D()
    testGauss2D()
    testGaussFilter()
