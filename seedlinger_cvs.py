import os
import sys
import warnings
import random

import cv2
import pyzed.sl as sl

sys.path.append('seedling_classifier')
sys.path.append('seedling_classifier/seedlingnet/modules')
sys.path.append('seedling_classifier/seedlingnet/modules/detectors')
sys.path.append('seedling_classifier/seedlingnet/modules/detectors/yolov7')
sys.path.append('seedling_classifier/seedlingnet/modules/classifiers')

from seedling_classifier.seedlingnet.modules.detector import Detector
from seedling_classifier.seedlingnet.modules.classifier import Classifier

global h_detector, v_detector, linear
h_detector = ""
v_detector = ""
linear = ""
h_wpath = '/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/' + \
        'detectors/weights/yolov7-hseed.pt'
v_wpath = '/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/' + \
        'detectors/weights/yolov7-vseed.pt'
dpath = '/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/' + \
        'detectors/weights/opt.yaml'

def call_yolo_predict(axis, img):
    global h_detector, v_detector
    if axis == "h":
        print("Getting predictions for horizontal poit of view")
        if h_detector == "":
            h_detector = Detector(
                'yolo7',
                weights=h_wpath,
                data=dpath,
                device='cuda:0'
            )
        predictions = h_detector.predict(img)
    elif axis == "v":
        print("Getting predictions for vertical poit of view")
        if v_detector == "":
            v_detector = Detector(
                'yolo7',
                weights=v_wpath,
                data=dpath,
                device='cuda:0'
            )
        predictions = v_detector.predict(img)
    else:
        raise Exception("Invalid axis inserted")

    print(f"Prediction result: {predictions}")
    return predictions

def get_type_of_seedling(hp, vp, vm):
    global linear
    tos = 0
    if hp is None or vp is None:
        tos = 1
    else:
        if len(hp) > 1 or len(vp) > 1:
            print("Issues with number of seedlings detected")
            print(f"Seedlings found in H view: {len(hp)}")
            print(f"Seedlings found in V view: {len(vp)}")
        else:
            print("Seedling detected properly")
            print(f"vp: {vp}")
            v_pred_mask = cv2.resize(
                vp.mask*255,
                (vm.shape[1], vm.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            h_pred_mask = hp.mask
            #load the model if necessary
            if linear == "":
                linear = Classifier('linear',CLASSIFIER_WEIGHTS)
            category = linear.predict(h_pred_mask, v_pred_mask)
            if category:
                tos = 3
            else:
                tos = 2
    print(f"Result of getting the type of seedling: {tos}")
    return tos

def print_prediction_info(predictions, img, top):
    if predictions is None:
        print(f"Point of view: {top}")
        print('Image Shape:',(img.shape), 'does not contains a seedling')
    else:
        for pred in predictions:
            x1, y1, x2, y2 = pred.bbox
            print(f'Image with shape: {img.shape}')
            print('contains a seedling with bounding box:')
            print(f'{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}')

def init_h_cam():
    #Instantiate the camera object
    cam_h = cv2.VideoCapture(0)
    #Warming up the camera, sort of
    for i in range(5):
        cam_h.read()
    # Check if the camera opened successfully
    if not cam_h.isOpened():
        raise Exception("Error: Unable to open X-axis camera.")

    return cam_h

def init_and_capture_v_cam():
    camera = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.4
    init_params.depth_maximum_distance = 0.8
    init_params.camera_image_flip = sl.FLIP_MODE.AUTO
    init_params.depth_stabilization = 1 
    runtime_params = sl.RuntimeParameters(
        confidence_threshold=50,
        enable_fill_mode=True,
        texture_confidence_threshold=200
    )

    err = camera.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(err)
        sys.exit()

    depth_map = sl.Mat()
    image = sl.Mat()

    assert camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS, \
            'La cámara Zed no esta ejecutandose correctamente,' + \
            'verificar su conexion'

    threshold_value = 130
    camera.retrieve_image(depth_map, sl.VIEW.DEPTH)
    numpy_depth_map = depth_map.get_data()
    gray = cv2.cvtColor(numpy_depth_map[:,:,0:3], cv2.COLOR_BGR2GRAY)
    gray = gray[290:890, 670:1550]
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    assert len(mask.shape) == 2, 'Verificar que se reciba una mascara' + \
                                 f'se recibe {mask.shape}'

    camera.retrieve_image(image, sl.VIEW.LEFT)
    numpy_image = image.get_data()
    img = numpy_image[:,:,0:3]
    img = img[290:890, 670:1550,:]

    assert len(img.shape) == 3, 'Verificar que se reciba una imagen RGB' + \
                                f', se recibe {img.shape}'

    img = cv2.bitwise_and(img,img,mask=mask)

    return img, mask

def run():
    #Camara horizontal
    cam_h = init_h_cam()
    # Capture an image
    ret, img = cam_h.read()
    # horizontal (x) axis prediction
    h_predictions = call_yolo_predict("h", img)
    # Print prediction info
    print_prediction_info(h_predictions, img, 'horizontal')

    #Camera vertical
    # Capture an imamge
    img, v_mask = init_and_capture_v_cam()
    # vertical (z) axis prediction
    v_predictions = call_yolo_predict("v", img)
    # Print prediction info
    print_prediction_info(v_predictions, img, 'vertical')

    #Get type of seedling
    tos = get_type_of_seedling(h_predictions, v_predictions, v_mask)

    return tos #random.randint(1,3)

if __name__ == "__main__":
    run()
