import os
import sys
import warnings
import random



from seedling_classifier.modules.detector import Detector
import cv2

def run():
    img_h = cv2.imread('gallery/horizontal.jpg')

    detector = Detector('yolo7', weights='weights/yolov7-hseed.pt', data='weights/opt.yaml', device='cuda:0')  # eg: 'yolov7' 'maskrcnn'
    predictions = detector.predict(img_h)

    for pred in predictions:
        x1, y1, x2, y2 = pred.bbox
        cv2.rectangle(img_h, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

    cv2.imshow('awd',img_h)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return random.randint(1,3)

if __name__ == "__main__":
    run()
