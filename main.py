import cv2 as cv
import numpy as np

import sys

def convolve(a, b):
    '''
    Computes convolution between `a` and `b` matrices where `b` represents a
    convolution kernel.
    '''
    result = a.copy()

    # Represents a pad width.
    width = (b.shape[0]-1) // 2

    pad = np.zeros((a.shape[0] + 2*width, a.shape[1] + 2*width))

    # Pad the `a` matrix.
    pad[width:-width, width:-width] = a

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            prod = np.multiply(pad[i:i + 2*width + 1,
                                   j:j + 2*width + 1], b)

            # Apply sum of the products.
            result[i, j] = np.sum(prod)

    return result

def mother_wavelet(t):
    '''
    Represents the mother wavelet with `t` parameter.
    '''
    return np.exp(-t*t / 2) - 0.5*np.exp(-t*t / 8)

def wavelet(t, tau, s):
    '''
    Represents a daughter wavelet generated by the mother wavelet with `t`,
    `tau` and `s` parameters.
    '''
    return 1/np.sqrt(s) * mother_wavelet(t-tau / s)

def wavelet_transform(image):
    '''
    Performs the wavelet transform of an `image`.
    '''
    pass

def main():
    '''
    The main function of the program.
    '''
    path = input('path: ')

    image = cv.imread(path, cv.IMREAD_UNCHANGED)

    cv.imshow('Initial Image', image)

    # Wait for user.
    cv.waitKey(0)

    sys.exit(0)

if __name__ == '__main__':
    main()
