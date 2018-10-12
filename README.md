# TensorFlow-Custom-Object-Detection
Object Detection using TensorFlow-Object-Detection_API

Object detection allows for the recognition, detection of multiple objects within an image.
It provides us a much better understanding of an image as a whole as opposed to just visual recognition.

Some real word applications of object detection include : self-driving car, tracking objects, face detection, etc.
<br/><br/><br/><br/>
## **Installation & Requirements:**
<br/><br/>
1. First things first:
> Download tensorflow/models repository from this [link](https://github.com/tensorflow/models).
<br/><br/>
2. Create new folder named TensorflowObjDet in `C:\`.
<br/><br/>
3. Extract downloaded `tensorflow/model` repository in this TensorflowObjDet folder.
<br/><br/>
4. Rename the models-master folder as models.
<br/><br/>
5. Download [this](https://github.com/niranjangavade5/TensorFlow-Custom-Object-Detection) repository & extract all files directly in 
`C:\TensorflowObjDet\models\research\object-detection`
<br/><br/>
6. You should atleast have _python 3.0 or above_ version & _Anaconda_ installed on your system.<br/>
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
> **Note:** _Before installing these packages you have to create a virtual enviroment for Tensorflow._
<br/><br/>
7. Open Anaconda Command prompt
<br/><br/>
8. Create virtual enviroment by executing :
```
conda create -n TensorflowVirtEnv
```
<br/><br/>
9. Activate created enviroment:
```
activate TensorflowVirtEnv
```
<br/><br/><br/>
10. Install required TensorFlow version in this enviroment by executing one of below commands:
- CPU version: 
```
pip install tensorflow
```
- GPU version:
```
pip install tensorflow-gpu
```
>**Note:**_If having any problem installing tensorflow, follow steps on this [link](https://www.tensorflow.org/install/)._
<br/><br/>
11. Run below command in same Command prompt to install all required packages:
```
conda install -r reqs.txt
```
<br/><br/>
12. Now change working directory by executing following command:
```
cd C:\TensorflowObjDet\models\research\object-detection\
```
<br/><br/>
13. Compile the Protobuf libraries by executing following command:
```
protoc protos/*.proto --python_out=.
```
<br/><br/>
14. Set enviroment path by executing following command:
```
set PATH=%PATH%;C:\TensorflowObjDet\models;C:\TensorflowObjDet\models\research;C:\TensorflowObjDet\models\research\slim
```
>**Note:** _This must be executed every time you open terminal._
<br/><br/>
15. Change directory by executing following command:
```
cd ..
```
<br/><br/>
16. Execute following commands one by one:
```
python setup.py build
python setup.py install
```
<br/><br/>
17. Change directory by executing following command:
```
cd object-detection
```
<br/><br/>
18. Delete temporary file `tp_delete_it.txt` in directories `train`,`test` & `inference_graph` 
<br/><br/><br/>
## **Setting train/test data & Creating `.csv` files**
<br/><br/>
1. Place train `.jpg` & their repective `.xml` files(labels) in train folder
<br/><br/>
2. Place test `.jpg` & their repective `.xml` files(labels) in test folder
> **Note :**_For creating .xml files(labels) you can use [this](https://github.com/tzutalin/labelImg) great open source image labelling tool. Repositories README.md file has information on installing & using it._
<br/>
> **Note :**_You can download train & test data from [here](https://www.kaggle.com/c/5408/download-all)._
<br/><br/>
4. Run the following command to create `train.csv` & `test.csv` files
```
python parse_xml_to_csv.py
```
<br/><br/><br/>
## **Creating TensorFlow Records**
> **Note:**_Input for TensorFlow model_
<br/><br/>
1. Open `create_tf_record.py` file & replace the label map in `class_text_to_int()` function on `line 33`, according to classes present in your images & save the file.
<br/><br/>
2. Run the following command to create `train.record`
```
python create_tf_record.py --csv_input=train_labels.csv --image_dir=train --output_path=train.record
```
<br/><br/>
3. Run the following command to create `test.record`
```
python create_tf_record.py --csv_input=test_labels.csv --image_dir=test --output_path=test.record
```
<br/><br/><br/>
## **Configuring label map**
<br/><br/>
1. Open `labelmap.pbtxt` file in `traning` folder
<br/><br/>
2. Change the label map present according to classes you have in your images & save the file
<br/><br/><br/>
## **Download Base Model**
<br/><br/>
1. Go to [model zoo](https://github.com/tensorflow/models/blob/99256cf470df6af16808eb0e49a8354d2f9beae2/research/object_detection/g3doc/detection_model_zoo.md) & download `faster_rcnn_inception_v2_coco` (or any other model of your choice).
<br/><br/>
2. Extract theis folder to `C:\TensorflowObjDet\models\research\object-detection\` this folder
<br/><br/><br/>
## **Configuring object detection training pipeline**
<br/><br/>
1. Open `samples\configs` folder and copy `faster_rcnn_inception_v2_pets.config` (or file for any other model that you use)
<br/><br/>
2. paste this file in `training` directory
<br/><br/>
3. Open this file & edit the following:
    - num_classes : to the number of classes you want to detect
    - fine_tune_checkpoint : set path to `C:/TensorflowModels/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt`(or to model.ckpt of any base model that you use)
    - input_path(train_input_reader) : in train_input_reader block set input_path to `C:/TensorflowModels/models/research/object_detection/train.record` file
    - label_map_path(train_input_reader) : in train_input_reader block set path to `C:/TensorflowModels/models/research/object_detection/training/labelmap.pbtxt`
    - num_examples : set the number to number of test images you will be using
    - input_path(eval_input_reader) : in eval_input_reader block set input_path to `C:/TensorflowModels/models/research/object_detection/test.record` file
    - label_map_path(eval_input_reader) : in eval_input_reader block set path to `C:/TensorflowModels/models/research/object_detection/training/labelmap.pbtxt`
    > **Note:**_Use forward slash(/) while setting these paths_
<br/><br/><br/>
## **Training**
<br/><br/>
1. Execute following command while in `C:/TensorflowModels/models/research/object_detection` directory
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config

```
> **Note:**_Initialization takes upto 30 secs before actual training starts_
<br/><br/>
2. Wait until loss is less than 0.05 consistently (can take upto hours, differs from model to model)
<br/><br/>
3. Once satisfied with loss value, press `control + c` to stop training
<br/><br/><br/>
## **Using Tensorboard(for graph visualization)**
<br/><br/>
1. In a new terminal navigate to `C:/TensorflowModels/models/research/object_detection` directory
<br/><br/>
2. Execute following command:
```
tensorboard --logdir=training
```
<br/><br/>
3. This command will generates a URL, without stopping the process copy that url and paste into browser
<br/><br/>
4. Tensorboard page will open up and you can visualize a lot of thing about currently being trained model, eg. different loss graphs.
<br/><br/><br/>
## **Exporting Inference graph**
<br/><br/>
1. Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the `\object_detection` folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered `.ckpt` file in the `training` folder. You might also change `--pipeline_config_path` parameters according to the model you use.
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph

```
<br/><br/>
2. This creates `frozen_inference_graph.pb` file in `inference_graph` directory
<br/><br/><br/>
### Model is ready for testing. You can write python scripts to test this model through webcam, images or video feeds as well.
