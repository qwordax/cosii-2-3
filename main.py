import cv2 as cv
import numpy as np

import os
import sys

SIZE = (512, 512)
'''
Specifies the size to show an image.
'''

def wavelet(x):
    '''
    Represents the mother wavelet with `x` parameter.
    '''
    return np.exp(-x*x / 2) - 0.5*np.exp(-x*x / 8)

def wavelet_transform(image):
    '''
    Performs the wavelet transform of an `image`.
    '''
    result = image.copy()

    return result

def main():
    '''
    The main function of the program.
    '''
    path = input('path: ')

    if not os.path.exists(path):
        print(f'error: \'{path}\' does not exists', file=sys.stderr)
        sys.exit(1)

    image = cv.imread(path, cv.IMREAD_COLOR)

    level = int(input('level: '))

    for l in range(level):
        print(f'scale = {4 ** (l+1)}')

        cv.imshow(f'Initial {l+1}', cv.resize(
            image, SIZE,
            cv.INTER_NEAREST
        ))

        result = image.copy()

        # Transform.
        for c in range(image.shape[0]):
            result[c] = wavelet_transform(image[c])

        # Obtain the approximation coefficient.
        image = result[:result.shape[0]//2, :result.shape[1]//2, :]

        cv.imshow(f'Transform {l+1}', cv.resize(
            result, SIZE,
            cv.INTER_NEAREST
        ))

        # Wait for user.
        cv.waitKey(0)
        cv.destroyAllWindows()

    sys.exit(0)

if __name__ == '__main__':
    main()
