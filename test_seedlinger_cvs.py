#import seedlinger_cvs

#if __name__ == "__main__":
#    seedlinger_cvs.run()
import os
import unittest
import sys
import warnings
import random
import time

import seedlinger_cvs

class test_x_z_runs(unittest.TestCase):
    def test_create_new_user(self):
        print("Test 1: test x and z axis detections and clasification")
        res = seedlinger_cvs.run()
        expected_res = 1
        self.assertEqual(res, expected_res)

if __name__ == '__main__':
    unittest.main()
