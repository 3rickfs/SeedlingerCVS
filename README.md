# **Seedlinger Computer Vision System**

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Tabla de contenidos</summary>
  <ol>
    <li>
      <a href="#Sobre el proyecto">Sobre el proyecto</a>
    </li>
    <li>
      <a href="#Accediendo al desarrollo">Accediendo al desarrollo</a>
      <ul>
        <li><a href="#prerequisitos">Prerequisitos</a></li>
        <li><a href="#instalacion">Instalacion</a></li>
      </ul>
    </li>
    <li><a href="#Forma de uso">Forma de uso</a></li>
    <li><a href="#Roadmap">Roadmap</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## Sobre el proyecto
Seedlinger Computer Vision System is a project developed to support the Picking Seedling Robot version II of the LABINM Robotics and Automation Laboratory. This system comprises of two detection YOLO8 models to infer the seedling object detection in vertical and horizontal images the robot lets it to gather through the process of seedling quality selection. To know the quality the system uses a custom deep learning model.

## Accediendo al desarrollo
The robot is located at the LABINM Robotics and Automation facilities, so the CVS is installed in a desktop PC dedicated only to the robot vision. This means the models were deployed locallyfor better performance in terms of prediction times. 

### Prerequisitos
To run the CVS module the following libraries are necessary:
* **OpenCV** to process images,
* **Pytorch** to run the detection and classification models,
* **Pymodbus** to connect the CVS with the robot's PLC.

## Forma de uso
To run the Seedlinger CVS:
1. Turn on the Robot desktop computer, choose the proper Ubuntu Linux version 
2. If any password is needed, just use the internet connection password of the lab.
3. Open a terminal and go to /home/robot/seedlinger/SeedlingerCVS
4. Run the following command
```
sudo bash server_calidad.sh
```

## Roadmap
- [ ] Add a UI to let the user monitor the performance of the seedlinger_cvs module
- [ ] Optimize the Seedlinger CVS to run as faster as possible.
- [ ] Test first 1.0.0 estable version with real artichoke seedling trays.
- [x] Launch first stable version.
