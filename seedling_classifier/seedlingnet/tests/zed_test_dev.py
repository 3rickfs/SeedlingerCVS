import sys
import cv2
import pyzed.sl as sl
import os

# Función para guardar la imagen
def save_image(image, folder, prefix, counter):
    filename = os.path.join(folder, f"{prefix}_{counter}.png")
    cv2.imwrite(filename, image)
    print(f"Imagen guardada en: {filename}")

output_folder_depth = "/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/scripts/capture_depth"
output_folder_rgb = "/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/scripts/capture_rgb"

os.makedirs(output_folder_depth, exist_ok=True)
os.makedirs(output_folder_rgb, exist_ok=True)

camera = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
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
rgb_image = sl.Mat()

counter_depth = 0
counter_rgb = 0

while True:
    if camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # Captura de imagen en profundidad
        camera.retrieve_image(depth_map, sl.VIEW.DEPTH)
        numpy_depth_map = depth_map.get_data()
        cv2.imshow('DEPTH', numpy_depth_map)
        
        # Captura de imagen en RGB
        camera.retrieve_image(rgb_image, sl.VIEW.LEFT)
        numpy_rgb_image = rgb_image.get_data()
        cv2.imshow('RGB', numpy_rgb_image)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): 
            break
        elif key & 0xFF == ord('s'):  # Guardar imágenes
            counter_depth += 1
            save_image(numpy_depth_map, output_folder_depth, "depth", counter_depth)
            
            counter_rgb += 1
            save_image(numpy_rgb_image, output_folder_rgb, "rgb", counter_rgb)

camera.close()
cv2.destroyAllWindows()
