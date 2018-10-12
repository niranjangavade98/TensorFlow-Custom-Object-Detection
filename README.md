# TensorFlow-Custom-Object-Detection
Object Detection using TensorFlow-Object-Detection_API

Object detection allows for the recognition, detection of multiple objects within an image.
It provides us a much better understanding of an image as a whole as opposed to just visual recognition.

Some real word applications of object detection include : self-driving car, tracking objects, face detection, etc.




## **Requirements & Installation:**


1. First things first:
>Download tensorflow/models repository from this [link](https://github.com/tensorflow/models).


2. Create new folder named TensorflowObjDet in C:\.


3. Extract downloaded `tensorflow/model` repository in this TensorflowObjDet folder.


4. Rename the models-master folder as models.


5. Download [this](https://github.com/niranjangavade5/TensorFlow-Custom-Object-Detection) repository & extract all files directly in 
`C:\TensorflowObjDet\models\research\object-detection`


6. You should atleast have _python 3.0 or above_ version & _Anaconda_ installed on your system.

**Required packages:**
```
protobuf
jupyter
opencv-python
matplotlib
pillow
pandas
numpy
lxml
```
>**Note:**_Before installing these packages you have to create a virtual enviroment for Tensorflow._


7. Open Anaconda Command prompt


8. Create virtual enviroment by executing :
```
conda create -n TensorflowVirtEnv
```


9. Activate created enviroment:
```
activate TensorflowVirtEnv
```


10. Install required TensorFlow version in this enviroment by executing one of below commands:
```
CPU version: `pip install tensorflow`
GPU version: `pip install tensorflow-gpu`
```
>**Note:**_If having any problem installing tensorflow, follow steps on this [link](https://www.tensorflow.org/install/)._


11. Run below command in same Command prompt to install all required packages:
```
conda install -r reqs.txt
```
  
  
12. Now change working directory by executing following command:
```
cd C:\TensorflowObjDet\models\research\object-detection\
```
  
  
13. Compile the Protobuf libraries by executing following command:
```
protoc protos/*.proto --python_out=.
```
  
  
14. Set enviroment path by executing following command:
```
set PATH=%PATH%;C:\TensorflowObjDet\models;C:\TensorflowObjDet\models\research;C:\TensorflowObjDet\models\research\slim
```
>**Note:** _This must be executed every time you open terminal._


