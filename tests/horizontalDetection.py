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

    YOLO7_WEIGHTS_H = '/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/detectors/weights/yolov7-hseed.pt'
    assert os.path.isfile(YOLO7_WEIGHTS_H) == True, f'Verificar la existencia del archivo {YOLO7_WEIGHTS_H}'

    TEST_IMAGE_H = '/home/robot/seedlinger/SeedlingerCVS/imagenes/horizontal/h-2024-06-01_14-12-14.510838_ll_A53_C0.jpg'
    assert os.path.isfile(TEST_IMAGE_H) == True, f'Verificar la existencia del archivo {TEST_IMAGE_H}'

    detector = Detector('yolo7', 
                        weights=YOLO7_WEIGHTS_H, 
                        data='seedling_classifier/seedlingnet/modules/detectors/weights/opt.yaml', 
                        device='cuda:0')  # eg: 'yolov7' 'maskrcnn'
    
    
    img = cv2.imread(TEST_IMAGE_H)
    predictions = detector.predict(img, threshold=0.4)
        
    if (predictions is None):
        print('Image Shape:',(img.shape), 'does not contains a seedling')
        return
    
    correct_predictions = []
    for pred in predictions:
        x1, y1, x2, y2 = pred.bbox
        if (y1 + y2)/2 > 280: continue

        correct_predictions.append(pred)
        print('Image Shape:',(img.shape), 'contains a seedling wich Bounding Box:', (int(x1), int(y1)), (int(x2), int(y2)))
    
    result = detector.model.plot_prediction(img, correct_predictions)
    cv2.imshow('Vertical Seedling',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return random.randint(1,3)
 

if __name__ == "__main__":
    run()
