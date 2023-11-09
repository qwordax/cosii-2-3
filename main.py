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

def wavelet_transform(data):
    '''
    Performs the wavelet transform of the `data`.
    '''
    pass

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

    level = int(input('level: '))

    for l in range(level):
        print(f'scale = {4 ** (l+1)}')

        cv.imshow(f'Initial {l+1}', cv.resize(
            image, size,
            cv.INTER_NEAREST
        ))

        result = image.copy()

        cv.imshow(f'Transform {l+1}', cv.resize(
            result, size,
            cv.INTER_NEAREST
        ))

        # Wait for user.
        cv.waitKey(0)
        cv.destroyAllWindows()

    sys.exit(0)

if __name__ == '__main__':
    main()
