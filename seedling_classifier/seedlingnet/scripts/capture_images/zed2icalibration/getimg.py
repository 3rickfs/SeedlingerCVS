import os
import pyzed.sl as sl
import cv2

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.AUTO  # Usar modo de video HD720 o HD1200, según el tipo de cámara.
    init_params.camera_fps = 30  

    # Abrir la cámara
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error al abrir la cámara: " + repr(err) + ". Salir del programa.")
        exit()

    cv2.namedWindow("Transmisión en tiempo real ZED", cv2.WINDOW_NORMAL)

    i = 0
    runtime_parameters = sl.RuntimeParameters()
    while True:
 
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

            image = sl.Mat()
            zed.retrieve_image(image, sl.VIEW.RIGHT)  # Capturar imagen con cámara derecha
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  
            image_data = image.get_data()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                image_path = os.path.join('/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/scripts/capture_images/zed2icalibration/images', f'imagen_{i}.png') 
                cv2.imwrite(image_path, image_data) 
                print(f"Imagen guardada en {image_path}")
                i += 1

            cv2.imshow("Transmisión en tiempo real ZED", image_data)

        else:
            print("Error al capturar la imagen")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
