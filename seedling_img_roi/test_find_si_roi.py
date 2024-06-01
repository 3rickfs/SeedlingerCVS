import os
import unittest
import sys

import find_si_roi

class test_find_si_roi(unittest.TestCase):
    def test_find_si_roi_1(self):
        print("Test 1: test finding the seedling image region of interest")
        res = find_si_roi.run()
        expected_res = {
            "x1": 20,
            "y1": 20,
            "x2": 20,
            "y2": 20
        }
        self.assertEqual(res, expected_res)

if __name__ == '__main__':
    unittest.main()

