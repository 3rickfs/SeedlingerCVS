import os
import sys
import warnings
import random

SEEDLING_CLASSIFIER_PATH  = '/home/robot/seedlinger/SeedlingerCVS'
sys.path.append(SEEDLING_CLASSIFIER_PATH)
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules'))
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules/detectors'))
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules/detectors/yolov7'))


from seedling_classifier.seedlingnet.modules.detector import Detector
import cv2

def run():

    YOLO7_WEIGHTS_V = '/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/detectors/weights/yolov7-vseed.pt'
    assert os.path.isfile(YOLO7_WEIGHTS_V) == True, f'Verificar la existencia del archivo {YOLO7_WEIGHTS_V}'

    detector = Detector('yolo7', 
                        weights=YOLO7_WEIGHTS_V, 
                        data='seedling_classifier/seedlingnet/modules/detectors/weights/opt.yaml', 
                        device='cuda:0')  # eg: 'yolov7' 'maskrcnn'
    
    TEST_IMAGE_V = '/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/detectors/gallery/vertical-rgb.jpg'
    assert os.path.isfile(TEST_IMAGE_V) == True, f'Verificar la existencia del archivo {TEST_IMAGE_V}'

    TEST_MASK_V = '/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/detectors/gallery/vertical-mask.jpg'
    assert os.path.isfile(TEST_IMAGE_V) == True, f'Verificar la existencia del archivo {TEST_IMAGE_V}'


    mask = cv2.imread(TEST_MASK_V,0)
    assert len(mask.shape) == 2, f'Verificar que se reciba una mascara'

    img = cv2.imread(TEST_IMAGE_V)
    assert len(img.shape) == 3, f'Verificar que la imagen recibida sea RGB'

    img = cv2.bitwise_and(img,img,mask=mask)
    predictions = detector.predict(img)
    
    if (predictions is None):
        print('Image Shape:',(img.shape), 'does not contains a seedling')
        return -1

    for pred in predictions:
        x1, y1, x2, y2 = pred.bbox
        print('Image Shape:',(img.shape), 'contains a seedling wich Bounding Box:', (int(x1), int(y1)), (int(x2), int(y2)))
    
    #result = detector.model.plot_prediction(img, predictions)
    #cv2.imshow('Vertical Seedling',result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return random.randint(1,3)
 

if __name__ == "__main__":
    run()
