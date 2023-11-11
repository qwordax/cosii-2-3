import cv2 as cv
import numpy as np

import os
import sys

HEIGHT = 500
'''
Specifies the height of the image to show.
'''

def wavelet(x):
    '''
    Represents the mother wavelet with `x` parameter.
    '''
    return np.exp(-x*x / 2) - 0.5*np.exp(-x*x / 8)

def low_pass(n):
    '''
    Returns the low pass filter kernel with even `n` parameter.
    '''
    result = np.zeros((n // 2, n))

    for i in np.arange(n // 2):
        result[i, 2*i] = 1
        result[i, 2*i + 1] = 1

    return 1/np.sqrt(2) * result

def high_pass(n):
    '''
    Returns the high pass filter kernel with even `n` parameter.
    '''
    result = np.zeros((n // 2, n))

    for i in np.arange(n // 2):
        result[i, 2*i] = 1
        result[i, 2*i + 1] = -1

    return 1/np.sqrt(2) * result

def wavelet_transform(data):
    '''
    Performs the wavelet transform of the `data`.
    '''
    if data.shape[0] % 2:
        data = np.concatenate((data, np.array([data[-1, :]])), axis=0)

    if data.shape[1] % 2:
        data = np.concatenate((data, np.array([data[:, -1]]).T), axis=1)

    approx = np.dot(data, low_pass(data.shape[1]).T)
    detail = np.dot(data, high_pass(data.shape[1]).T)

    a = np.dot(low_pass(approx.shape[0]), approx)
    h = np.dot(high_pass(approx.shape[0]), approx)
    v = np.dot(low_pass(detail.shape[0]), detail)
    d = np.dot(high_pass(detail.shape[0]), detail)

    # Normalize coefficients.
    a = a / np.max(a) * 255.0
    h = h / np.max(h) * 255.0
    v = v / np.max(v) * 255.0
    d = d / np.max(d) * 255.0

    return np.uint8(a), np.uint8(h), np.uint8(v), np.uint8(d)

def main():
    '''
    The main function of the program.
    '''
    path = input('path: ')

    if not os.path.exists(path):
        print(f'error: \'{path}\' does not exists', file=sys.stderr)
        sys.exit(1)

    image = cv.imread(path, cv.IMREAD_GRAYSCALE)

    # The size of the image to show.
    size = (int(HEIGHT * image.shape[1]/image.shape[0]), HEIGHT)

    # The size of the coefficients to show.
    coeff_size = (size[0] // 2, size[1] // 2)

    level = int(input('level: '))

    for l in np.arange(level):
        cv.imshow(f'Initial {l+1}', cv.resize(image, size))

        # Transform the image.
        a, h, v, d = wavelet_transform(image)

        cv.imshow(f'Approximation {l+1}', cv.resize(a, coeff_size))
        cv.imshow(f'Horizontal {l+1}', cv.resize(h, coeff_size))
        cv.imshow(f'Vertical {l+1}', cv.resize(v, coeff_size))
        cv.imshow(f'Diagonal {l+1}', cv.resize(d, coeff_size))

        print(f'scale = {4 ** (l+1)}')

        # Wait for user.
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Update the image for the next iteration.
        image = a

    sys.exit(0)

if __name__ == '__main__':
    main()
