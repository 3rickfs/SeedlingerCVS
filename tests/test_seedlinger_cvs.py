import os
import unittest
import sys
import warnings
import random
import time

SEEDLING_CLASSIFIER_PATH  = '/home/robot/seedlinger/SeedlingerCVS'
sys.path.append(SEEDLING_CLASSIFIER_PATH)

import seedlinger_cvs 

class distribution_tests(unittest.TestCase):
    def test_create_new_user(self):
        print("Test 1: Create new user in Brain")
        res = seedlinger_cvs.run()
        expected_res = 1
        self.assertEqual(res, expected_res)

if __name__ == '__main__':
    unittest.main()
