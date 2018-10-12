# TensorFlow-Custom-Object-Detection
Object Detection using TensorFlow-Object-Detection_API

Object detection allows for the recognition, detection of multiple objects within an image.
It provides us a much better understanding of an image as a whole as opposed to just visual recognition.

Some real word applications of object detection include : self-driving car, tracking objects, face detection, etc.

**Requirements & Installation:**

1. First things first:
> Download tensorflow/models repository from this [link](https://github.com/tensorflow/models).

2. Create new folder named TensorflowObjDet.

3. Extract downloaded tensorflow/model repository in this TensorflowObjDet.

4. Rename the models-master folder as models.

5. Download [this]() repository & extract all files directly in 
C:\TensorflowObjDet\models\research\object-detection 

You should atleast have python 3.0 or above version & Anaconda installed on your system.

Required packages:
-c anaconda protobuf
jupyter
opencv-python
matplotlib
pillow
pandas
numpy
lxml

Before installing these packages you have to create a virtual enviroment for Tensorflow.

Open Anaconda Command prompt

Create virtual enviroment by executing :
conda create -n TensorflowVirtEnv

Activate created enviroment:
activate TensorflowVirtEnv

Install TensorFlow in this enviroment by executing one of below commands:
CPU version: pip install tensorflow 
GPU version: pip install tensorflow-gpu

Run below command in same Command prompt to install all required packages:
conda install -r reqs.txt

Now change working directory by executing following command:
cd C:\TensorflowObjDet\models\research\object-detection\

Compile the Protobuf libraries by executing following command:
protoc protos/*.proto --python_out=.

Set enviroment by executing following command:
set PATH=%PATH%;C:\TensorflowObjDet\models;C:\TensorflowObjDet\models\research;C:\TensorflowObjDet\models\research\slim
Note: This must be executed every time you open terminal.


