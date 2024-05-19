import os
import sys
import warnings
import random

sys.path.append('seedling_classifier')
sys.path.append('seedling_classifier/seedlingnet/modules')
sys.path.append('seedling_classifier/seedlingnet/modules/detectors')
sys.path.append('seedling_classifier/seedlingnet/modules/detectors/yolov7')


from seedling_classifier.seedlingnet.modules.detector import Detector

import cv2

def run():

    cam_h = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cam_h.isOpened():
        print("Error: Unable to open camera.")
        return
    
    #img_h = cv2.imread('seedling_classifier/seedlingnet/gallery/horizontal.jpg')

    detector = Detector('yolo7', 
                        weights='seedling_classifier/seedlingnet/modules/detectors/weights/yolov7-hseed.pt', 
                        data='seedling_classifier/seedlingnet/modules/detectors/weights/opt.yaml', 
                        device='cuda:0')  # eg: 'yolov7' 'maskrcnn'
    
    while True:
        # Capture frame-by-frame
        ret, img_h = cam_h.read()
        
        predictions = detector.predict(img_h)

        if not(predictions is None):
            for pred in predictions:
                x1, y1, x2, y2 = pred.bbox
                cv2.rectangle(img_h, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

        cv2.imshow('awd',img_h)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return random.randint(1,3)

if __name__ == "__main__":
    run()
