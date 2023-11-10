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

def kernel(n):
    '''
    Returns the filter kernel with even `n` parameter.
    '''
    result = np.zeros((n, n))

    for i in np.arange(n // 2):
        result[i, 2*i] = 1
        result[i, 2*i + 1] = 1
        result[i + n//2, 2*i] = 1
        result[i + n//2, 2*i + 1] = -1

    return result

def low_pass_filter(data):
    '''
    Represents the low pass filter based on the mother wavelet.
    '''
    return data[:, ::2].copy()

def high_pass_filter(data):
    '''
    Represents the high pass filter based on the mother wavelet.
    '''
    return data[:, ::2].copy()

def wavelet_transform(data):
    '''
    Performs the wavelet transform of the `data`.
    '''
    coeff_a = list()
    coeff_h = list()
    coeff_v = list()
    coeff_d = list()

    for c in np.arange(data.shape[2]):
        approx = low_pass_filter(data[:, :, c])
        detail = high_pass_filter(data[:, :, c])

        a = low_pass_filter(approx.T).T
        h = low_pass_filter(detail.T).T
        v = high_pass_filter(approx.T).T
        d = high_pass_filter(detail.T).T

        coeff_a.append(a.reshape(a.shape[0], a.shape[1], 1))
        coeff_h.append(h.reshape(h.shape[0], h.shape[1], 1))
        coeff_v.append(v.reshape(v.shape[0], v.shape[1], 1))
        coeff_d.append(d.reshape(d.shape[0], d.shape[1], 1))

    return (
        np.concatenate(coeff_a, axis=2),
        np.concatenate(coeff_h, axis=2),
        np.concatenate(coeff_v, axis=2),
        np.concatenate(coeff_d, axis=2)
    )

def main():
    '''
    The main function of the program.
    '''
    path = input('path: ')

    if not os.path.exists(path):
        print(f'error: \'{path}\' does not exists', file=sys.stderr)
        sys.exit(1)

    image = cv.imread(path, cv.IMREAD_COLOR)

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
        coeff_a, coeff_h, coeff_v, coeff_d = wavelet_transform(image)

        cv.imshow(f'Approximation {l+1}', cv.resize(
            coeff_a, coeff_size,
            cv.INTER_NEAREST
        ))

        cv.imshow(f'Horizontal {l+1}', cv.resize(
            coeff_h, coeff_size,
            cv.INTER_NEAREST
        ))

        cv.imshow(f'Vertical {l+1}', cv.resize(
            coeff_v, coeff_size,
            cv.INTER_NEAREST
        ))

        cv.imshow(f'Diagonal {l+1}', cv.resize(
            coeff_d, coeff_size,
            cv.INTER_NEAREST
        ))

        # Wait for user.
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Update the image for the next iteration.
        image = coeff_a

    sys.exit(0)

if __name__ == '__main__':
    main()
