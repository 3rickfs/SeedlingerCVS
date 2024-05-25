import os
import sys
import warnings
import random
import time

SEEDLING_CLASSIFIER_PATH  = '/home/robot/seedlinger/SeedlingerCVS'
sys.path.append(SEEDLING_CLASSIFIER_PATH)
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules'))
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules/detectors'))
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules/detectors/yolov7'))


from seedling_classifier.seedlingnet.modules.detector import Detector
import cv2
import numpy as np

def run(show):
    cam_h = cv2.VideoCapture(0)

    for i in range(5):
        cam_h.read()

    assert cam_h.isOpened() == True , f'Error: No se recibe imagen de la cámara horizontal'

    YOLO7_WEIGHTS_H = '/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/detectors/weights/yolov7-hseed.pt'
    assert os.path.isfile(YOLO7_WEIGHTS_H) == True, f'Verificar la existencia del archivo {YOLO7_WEIGHTS_H}'

    TEST_IMAGE_H = '/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/detectors/gallery/horizontal.jpg'
    assert os.path.isfile(TEST_IMAGE_H) == True, f'Verificar la existencia del archivo {TEST_IMAGE_H}'

    detector = Detector('yolo7', 
                        weights=YOLO7_WEIGHTS_H, 
                        data='seedling_classifier/seedlingnet/modules/detectors/weights/opt.yaml', 
                        device='cuda:0')  # eg: 'yolov7' 'maskrcnn'


    ret, img_h = cam_h.read()
    #img_h = cv2.rotate(img_h, cv2.ROTATE_90_COUNTERCLOCKWISE)

    predictions = detector.predict(img_h)

    pred_mask = np.array([])
    pred_bbox = np.array([])
    if (predictions is None):
        print('Image Shape:',(img_h.shape), 'does not contains a seedling')
    else:
        for pred in predictions:
            x1, y1, x2, y2 = pred.bbox
            result = detector.model.plot_prediction(img_h, predictions)
            print('Image Shape:',(img_h.shape), 'contains a seedling at Bounding Box:', (int(x1), int(y1)), (int(x2), int(y2)))
            print('Press any Key to close this window')
        pred_mask = pred.mask
        pred_bbox = pred.bbox

        cv2.imshow('horizontal view',img_h)
        cv2.imshow('horizontal mask',pred.mask*255)
        cv2.waitKey(0)
        cam_h.release()
        cv2.destroyAllWindows()

    if not(show):
        return (pred.bbox, pred.mask)
    
    return (pred_bbox, pred_mask)
    
if __name__ == "__main__":
    run(show=True)

