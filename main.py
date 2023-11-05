import cv2 as cv

import sys

def main():
    '''
    The main function of the program.
    '''
    path = input('path: ')

    image = cv.imread(path)

    cv.imshow('Initial Image', image)

    # Waiting for user.
    cv.waitKey(0)

    sys.exit(0)

if __name__ == '__main__':
    main()
