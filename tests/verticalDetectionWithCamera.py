import os
import sys
import warnings
import random
import pyzed.sl as sl

SEEDLING_CLASSIFIER_PATH  = '/home/robot/seedlinger/SeedlingerCVS'
sys.path.append(SEEDLING_CLASSIFIER_PATH)
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules'))
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules/detectors'))
sys.path.append(os.path.join(SEEDLING_CLASSIFIER_PATH,'seedling_classifier/seedlingnet/modules/detectors/yolov7'))




from seedling_classifier.seedlingnet.modules.detector import Detector
import cv2

def run(show):

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
    runtime_params = sl.RuntimeParameters(confidence_threshold=50,  enable_fill_mode=True, texture_confidence_threshold=200)

    err = camera.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(err)
        sys.exit()

    depth_map = sl.Mat()
    image = sl.Mat()

    assert camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS, 'La cámara Zed no esta ejecutandose correctamente, verificar su conexion'
    
    threshold_value = 130
    camera.retrieve_image(depth_map, sl.VIEW.DEPTH)
    numpy_depth_map = depth_map.get_data()
    gray = cv2.cvtColor(numpy_depth_map[:,:,0:3], cv2.COLOR_BGR2GRAY)
    gray = gray[290:890, 670:1550]
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    assert len(mask.shape) == 2, f'Verificar que se reciba una mascara, se recibe {mask.shape}'
    
    camera.retrieve_image(image, sl.VIEW.LEFT)
    numpy_image = image.get_data()
    img = numpy_image[:,:,0:3]
    complete_image = img
    img = img[290:890, 670:1550,:]

    assert len(img.shape) == 3, f'Verificar que se reciba una imagen RGB, se recibe {img.shape}'
    
    img = cv2.bitwise_and(img,img,mask=mask)


    YOLO7_WEIGHTS_V = '/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/detectors/weights/yolov7-vseed.pt'
    assert os.path.isfile(YOLO7_WEIGHTS_V) == True, f'Verificar la existencia del archivo {YOLO7_WEIGHTS_V}'
    
    detector = Detector('yolo7', 
                        weights=YOLO7_WEIGHTS_V, 
                        data='seedling_classifier/seedlingnet/modules/detectors/weights/opt.yaml', 
                        device='cuda:0')  # eg: 'yolov7' 'maskrcnn'
    
    predictions = detector.predict(img)
    
    if (predictions is None):
        print('Image Shape:',(img.shape), 'does not contains a seedling')
        return (None, None)
    else:
        for pred in predictions:
            x1, y1, x2, y2 = pred.bbox
            result = detector.model.plot_prediction(img, predictions)
            print('Image Shape:',(img.shape), 'contains a seedling at Bounding Box:', (int(x1), int(y1)), (int(x2), int(y2)))
            print('Press any Key to close this qq')

    mask_shape = mask.shape
    if not(show):
        return (pred.bbox, cv2.resize(pred.mask*255, (mask_shape[1],mask_shape[0]), interpolation=cv2.INTER_LINEAR))

    
    print(mask_shape)
    cv2.imshow('horizontal view',img)
    cv2.imshow('depth map',gray)
    #cv2.imshow('complete_img',complete_image)
    cv2.imshow('mask', cv2.resize(pred.mask*255, (mask_shape[1],mask_shape[0]), interpolation=cv2.INTER_LINEAR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (pred.bbox, pred.mask)
 

if __name__ == "__main__":
    run(show=True)
