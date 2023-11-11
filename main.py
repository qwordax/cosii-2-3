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
        result[i, 2*i] = -1
        result[i, 2*i + 1] = 1

    return 1/np.sqrt(2) * result

def wavelet_transform(data):
    '''
    Performs the wavelet transform of the `data`.
    '''
    a = data.copy()
    h = data.copy()
    v = data.copy()
    d = data.copy()

    return a, h, v, d

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
        print(f'scale = {4 ** (l+1)}')

        cv.imshow(f'Initial {l+1}', cv.resize(
            image, size,
            cv.INTER_NEAREST
        ))

        # Transform the image.
        a, h, v, d = wavelet_transform(image)

        cv.imshow(f'Approximation {l+1}', cv.resize(
            a, coeff_size,
            cv.INTER_NEAREST
        ))

        cv.imshow(f'Horizontal {l+1}', cv.resize(
            h, coeff_size,
            cv.INTER_NEAREST
        ))

        cv.imshow(f'Vertical {l+1}', cv.resize(
            v, coeff_size,
            cv.INTER_NEAREST
        ))

        cv.imshow(f'Diagonal {l+1}', cv.resize(
            d, coeff_size,
            cv.INTER_NEAREST
        ))

        # Wait for user.
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Update the image for the next iteration.
        image = a

    sys.exit(0)

if __name__ == '__main__':
    main()
