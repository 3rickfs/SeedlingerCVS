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

    predictions = detector.predict(img_h, threshold=0.5)

    if (predictions is None):
        print('Image Shape:',(img_h.shape), 'does not contains a seedling')
        return (None, None)
    else:
        for pred in predictions:
            x1, y1, x2, y2 = pred.bbox
            #area = (x2-x1)*(y2-y1)
            #if (x1 + x2)//2 > 250 or area < ((312-255)*(281-201)):
            #    continue
            result = detector.model.plot_prediction(img_h, predictions)
            print('Image Shape:',(img_h.shape), 'contains a seedling at Bounding Box:', (int(x1), int(y1)), (int(x2), int(y2)))
            print('Press any Key to close this window')

    
    if not(show):
        return (pred.bbox, pred.mask)
    
    cv2.imshow('horizontal view',img_h)
    cv2.imshow('horizontal mask',pred.mask*255)
    cv2.waitKey(0)
    cam_h.release()
    cv2.destroyAllWindows()

    return (pred.bbox, pred.mask)
    
if __name__ == "__main__":
    run(show=True)

