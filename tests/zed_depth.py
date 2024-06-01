import sys
import cv2
import pyzed.sl as sl
import os

# Función para guardar la imagen
def save_depth_image(depth_map, folder):
    filename = os.path.join(folder, "depth_image.png")
    cv2.imwrite(filename, depth_map)
    print(f"Imagen de profundidad guardada en: {filename}")

output_folder = "/home/robot/seedlinger/SeedlingerCVS/weights" 

os.makedirs(output_folder, exist_ok=True)

camera = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_minimum_distance = 0.4
init_params.depth_maximum_distance = 0.8
init_params.camera_image_flip = sl.FLIP_MODE.AUTO
init_params.depth_stabilization = 1 
runtime_params = sl.RuntimeParameters(confidence_threshold=50, texture_confidence_threshold=200)

err = camera.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(err)
    sys.exit()

depth_map = sl.Mat()
image = sl.Mat()

while True:
    if camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        camera.retrieve_image(depth_map, sl.VIEW.DEPTH)
        numpy_depth_map = depth_map.get_data()
        
        camera.retrieve_image(image, sl.VIEW.LEFT)
        numpy_image = image.get_data()

        threshold_value = 128
        gray = cv2.cvtColor(numpy_depth_map[:,:,0:3], cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        img = numpy_image[:,:,0:3]
        print(mask.shape)
        print(img.shape)
        cv2.imshow('DEPTH', mask)
        cv2.imshow('IMAGE RGB', numpy_image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): 
            break
        elif key & 0xFF == ord('s'):  # Guardar imagen
            save_depth_image(numpy_depth_map, output_folder)

camera.close()
cv2.destroyAllWindows()
