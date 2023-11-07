import cv2 as cv

import sys

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

    # Waiting for user.
    cv.waitKey(0)

    sys.exit(0)

if __name__ == '__main__':
    main()
