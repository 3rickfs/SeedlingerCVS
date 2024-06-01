import os
import cv2

def run():
    print("Running the finding seedling image region of interest")
    siroi = ""
    img_files = [f for f in os.listdir("/home/dev-1/dev/SeedlingerRobotCVSytem/imagenes/horizontal") if os.path.isfile(f)]

    for f in img_files:
        print(f)

    return siroi


if __name__ == '__main__':
    run()
