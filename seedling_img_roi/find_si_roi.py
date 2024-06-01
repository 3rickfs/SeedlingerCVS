import os
import cv2
import sys

SEEDLING_CLASSIFIER_PATH  = '/home/robot/seedlinger/SeedlingerCVS'
sys.path.append(SEEDLING_CLASSIFIER_PATH)
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules'))
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules/detectors'))
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules/detectors/yolov7'))

from seedling_classifier.seedlingnet.modules.detector import Detector

verbose = False
h_wpath = '/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/'

def get_si_bboxs(ifps):
    print("Getting the seedling image bbox list")

    bbox_l = []
    for ifp in ifps:
        fp = SEEDLING_CLASSIFIER_PATH + "/imagenes/horizontal/"
        imgfp = fp + ifp
        img = cv2.imread(imgfp, cv2.IMREAD_COLOR)

        h_detector = Detector(
            'yolo7',
            weights=h_wpath,
            data=dpath,
            device='cuda:0'
        )

        predictions = h_detector.predict(img)

        pred_mask = np.array([])
        pred_bbox = np.array([])
        if (predictions is None):
            print('Image Shape:',(img_h.shape), 'does not contains a seedling')
        else:
            for pred in predictions:
                x1, y1, x2, y2 = pred.bbox
                result = detector.model.plot_prediction(img_h, predictions)
                print('Image Shape:',(img_h.shape), 'contains a seedling at Bounding Box:', (int(x1), int(y1)), (int(x2), int(y2)))
                bbox_l.append(pred.bbox)

    return bbox_l

def run():
    print("Running the finding seedling image region of interest")
    siroi = ""
    img_files = [f for f in os.listdir("/home/robot/seedlinger/SeedlingerCVS/imagenes/horizontal")]

    if verbose:
        print("printing img file names")
        for f in img_files:
            print(f)

    #apply detection
    sibboxs = get_si_bboxs(img_files)
    print(f"Seedling image bounding box list: {sibboxs}")

    return siroi

if __name__ == '__main__':
    run()
