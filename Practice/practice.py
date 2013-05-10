'''
Created on Sep 23, 2012

@author: will
'''
from numpy.core.numeric import arange, outer
from numpy.ma.extras import dot

if __name__ == '__main__':
    x = arange(10)
    print x
    print dot(x,x)
    print outer(x,x)