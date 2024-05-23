import os
import sys
import warnings
import random

SEEDLING_CLASSIFIER_PATH  = '/home/robot/seedlinger/SeedlingerCVS'
sys.path.append(SEEDLING_CLASSIFIER_PATH)
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules'))
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules/classifiers'))


from seedling_classifier.seedlingnet.modules.classifier import Classifier
import cv2

def run():

    CLASSIFIER_WEIGHTS = '/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/classifiers/weights/linearModel.pt'
    assert os.path.isfile(CLASSIFIER_WEIGHTS) == True, f'Verificar la existencia del archivo {CLASSIFIER_WEIGHTS}'

    horizontalMask = cv2.imread('/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/classifiers/gallery/horizontal.jpg',0)
    assert len(horizontalMask.shape) == 2, f'Verificar que se reciba una mascara'

    verticalMask = cv2.imread('/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/classifiers/gallery/vertical.jpg',0)
    assert len(verticalMask.shape) == 2, f'Verificar que se reciba una mascara'

    linear = Classifier('linear',CLASSIFIER_WEIGHTS)
    category = linear.predict(horizontalMask, verticalMask)
    print('seedling-category:   good:',category, ' bad:', not category)

    return random.randint(1,3)
 

if __name__ == "__main__":
    run()
