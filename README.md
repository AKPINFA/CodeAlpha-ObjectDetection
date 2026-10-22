 Object Detection and Tracking Using YOLO11



 Project Overview



This project implements real time object detection and tracking using Python, YOLO11, OpenCV, and Deep SORT. It demonstrates how computer vision techniques can be used to detect objects in video streams and track their movements across consecutive frames.



The project was developed as part of the CodeAlpha Internship Program to gain practical experience in computer vision and intelligent video analysis.



 Objectives



The main objective is to build a computer vision system capable of detecting objects from a live webcam feed and tracking their movements in real time.



The project also aims to explore object detection models, video processing, and the integration of computer vision libraries in Python.



 Features



Real time object detection using the YOLO11 model.



Webcam integration for live video capture.



Object tracking across consecutive video frames.



Visual display of detected objects and tracking results.



Modular Python scripts for testing, detection, and tracking.



 Technologies Used



Python for application development.



YOLO11 for object detection.



OpenCV for image processing and webcam integration.



Deep SORT for object tracking.



Git and GitHub for version control and project hosting.



 Project Structure



```text

CodeAlpha-ObjectDetection/

&#x20;   camera\_test.py

&#x20;   detect.py

&#x20;   track.py

&#x20;   yolo11n.pt

&#x20;   requirements.txt

&#x20;   README.md

```



 Requirements



Python 3.10 or a compatible version supported by the selected dependencies.



A working webcam for real time detection and tracking.



Pip for Python package management.



 Installation



Clone the repository:



```bash

git clone https://github.com/AKPINFA/CodeAlpha-ObjectDetection.git

cd CodeAlpha-ObjectDetection

```



Create a virtual environment:



```bash

python -m venv venv

```



Activate the environment on Windows PowerShell:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



Install the project dependencies:



```bash

pip install -r requirements.txt

```



Usage



To test webcam access, run:



```bash

python camera\_test.py

```



To run object detection, execute:



```bash

python detect.py

```



To run object detection and tracking, execute:



```bash

python track.py

```



Make sure your webcam is connected and available before running the scripts.



 Model



The project uses the YOLO11 Nano model, provided in the `yolo11n.pt` file. This model offers a lightweight approach to object detection and is suitable for experimentation with real time computer vision applications.



 Learning Outcomes



This project provides practical experience with real time video processing, object detection, object tracking, Python development, and the use of computer vision libraries.



 Internship



Developed as part of the CodeAlpha Internship Program.



 Author







GitHub: https://github.com/AKPINFA



Project Repository: https://github.com/AKPINFA/CodeAlpha-ObjectDetection



