import sys
import os
SEEDLING_CLASSIFIER_PATH  = '/home/robot/seedlinger/SeedlingerCVS'
sys.path.append(SEEDLING_CLASSIFIER_PATH)
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules'))
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules/detectors'))
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules/detectors/yolov7'))
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules/classifiers'))

import warnings
import random
import time

import cv2
import matplotlib.pyplot as plt
from horizontalDetectionWithCamera import run as getHorizontalMask
from verticalDetectionWithCamera import run as getVerticalMask
from seedling_classifier.seedlingnet.modules.classifier import Classifier

def run():
    bboxh, horizontalMask = getHorizontalMask(show=True)
    bboxv, verticalMask = getVerticalMask(show=True)

    CLASSIFIER_WEIGHTS = '/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/classifiers/weights/linearModel.pt'
    assert os.path.isfile(CLASSIFIER_WEIGHTS) == True, \
            f'Verificar la existencia del archivo {CLASSIFIER_WEIGHTS}'
    assert len(horizontalMask.shape) == 2, \
            f'No seedling detected, horizontal. Or check h-camera'
    assert len(verticalMask.shape) == 2, \
            f'No seedling detected, vertical. Or check v-camera'

    linear = Classifier('linear',CLASSIFIER_WEIGHTS)
    category = linear.predict(horizontalMask, verticalMask)
    print('seedling-category:   good:',category, ' bad:', not category)
    return


if __name__ == '__main__':
    run()

