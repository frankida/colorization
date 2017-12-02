import cv2
import numpy as np
import scipy as sp
import unittest
import os

import colorization as col


SOURCE_FOLDER = os.path.abspath(os.path.join(os.curdir, 'videos', 'source'))
EXTS = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg',
        '.jpe', '.jp2', '.tiff', '.tif', '.png']


class Final_project(unittest.TestCase):

    def setUp(self):


        self.images = 0

    def test1(self):
        print "hello world"
        print self.images

if __name__ == '__main__':
    unittest.main(verbosity=2)
