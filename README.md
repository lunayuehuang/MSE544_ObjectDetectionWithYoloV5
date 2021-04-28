# MSE544 Object Detection With YoloV5

## Part 1 How to train and run YoloV5 on local machines

### Step A. Get YoloV5 an set up python environment

Open your terminal and make a new directory named ```MSE544_yolo_training```(or names that you like). Switch into the directory and then clone the yolo repository from GitHub:
```
git clone https://github.com/ultralytics/yolov5
```

Make a new conda enviroment with python 3.8 or later and install the required packages
```
# create conda environment
conda create -n yolov5 python=3.8 jupyter notebook
conda activate yolov5

cd yolov5 
pip install -r requirements.txt  # packages required by yolov5
pip install sklearn scikit-image azureml-core # other packages used in this tutorial 
```

### Step B. Prepare Yolo labels

Locate the repository (https://github.com/lunayuehuang/Mse544-CustomVision) from Monday's class or clone it if you haven't done so. In the rest of this tutorial, the path of Monday's repository will refer as ```<path-to-Mse544-CustomVision>```, which will be replaced by the real path on your computer. 

Go out of the ```yolov5``` folder (back to ```MSE544_yolo_training```) and copy a file ```util.py``` from ```<path-to-Mse544-CustomVision>``` to current one.
```
cd <path-to-MSE544_yolo_training>
cp <path-to-Mse544-CustomVision>/util.py .
```

Now, start a fresh jupyter notebook, named by ```molecule_detection_yolo_training.ipynb```. In the first cell import the utility functions:
```
from util import labeledImage, normalize_coordinates, convert_to_yolo_format
from sklearn.model_selection import train_test_split
import os, shutil, yaml
```

Then use the helper class we have from Monday ```labeledImage``` to load all the labels that produced by ImageJ:
```
source_images_dir = '<path-to-Mse544-CustomVision>/molecules/'
source_labels_dir = '<path-to-Mse544-CustomVision>/molecules/labels/'

labeled_images = []
tag = 'molecule' 

for file in os.listdir(source_images_dir):
    # find all jpeg file and it's ImageJ label
    if file.endswith(".jpeg"):
        image_path = os.path.join(source_images_dir, file)
        label_path = os.path.join(source_labels_dir, file.split('.')[0] + '.txt')
        labeled_images.append(labeledImage(image_path))
        labeled_images[-1].add_labels_from_file(tag, label_path)
```

In next cell. let's split the labled images as training, validation and testing. Normally the ratio of them is 7:2:1.
```
train_and_val_set, test_set = train_test_split(labeled_images, test_size=0.1)
train_set, val_set = train_test_split(train_and_val_set, test_size=(2/9))

len(train_set), len(val_set), len(test_set)
```

The output of this cell will show the size of training, valiation and testing set. Particularly, for this example, it's
```
(35, 10, 5)
```

Before pouring the images and labels, let's create our data directory hierarchy as
```
|---image_data
    |---training.yaml
    |---train
        |---images
        |---labels
    |---val
        |---images
        |---labels
    |---test
        |---images
        |---lables
```
where ```training.yaml``` is a configuration file for yolo that stores all the parameters information needed by yolov5. Run the following code in your notebook will produce such data structure, and ```molecule_images``` is used for the name of this image dataset.

```
# making directories
output_dir = './molecule_images'

if not os.path.exists(output_dir): os.mkdir(output_dir)

train_dir = os.path.join(output_dir, 'train') 
val_dir   = os.path.join(output_dir, 'val') 
test_dir  = os.path.join(output_dir, 'test') 

for d in [train_dir, val_dir, test_dir]:
    if not os.path.exists(d): os.mkdir(d)
    
    images_sub_dir = os.path.join(d, 'images')
    labels_sub_dir = os.path.join(d, 'labels')
    
    for sub_dir in [images_sub_dir, labels_sub_dir]:
        if not os.path.exists(sub_dir): os.mkdir(sub_dir)
```
Now, it's ready to copy over all the images file and convert all the ImageJ labels into yolo format:
```
# make unified yolo tags 
tags = [tag]

# zip the dataset
dataset = [(train_dir, train_set),(val_dir, val_set),(test_dir, test_set)]

for d, s in dataset:
    images_sub_dir = os.path.join(d, 'images')
    labels_sub_dir = os.path.join(d, 'labels')

    # copy over the images
    for img in s:
        shutil.copyfile(img.path, os.path.join(images_sub_dir, img.name))
    
    # covert ImageJ labels to yolo format and save it to labels_sub_dir
    convert_to_yolo_format(s, labels_sub_dir, tags)
```
The last step is to generate a configuration file for training:
```
# generate yolo yaml file
yolo_yaml = os.path.join(output_dir, 'molecule_detection_yolov5.yaml')

with open(yolo_yaml, 'w') as yamlout:
    yaml.dump(
        {'train': train_dir,
         'val': val_dir,
         'nc': len(tags),
         'names': tags},
        yamlout,
        default_flow_style=None,
        sort_keys=False
    )
```
### Step C. Training the YoloV5 modle on local machines    
With all the label prepared, you can try to train a few epoch on your local machine by simpily go into ```yolov5``` folder from your notebook:
```
%cd yolov5
```
and then run the training command in next cell:
```
!python train.py --img 640 --batch 16 --epochs 1 --data ../molecule_images/molecule_detection_yolov5.yaml --weights yolov5s.pt
```
As you might noticed that, training yolov5 model on your local machine is very slow, where the GPU training cluster on Azure Machine Learing could be used to speed up our training. 

The logs of your training is will be located at ```yolov5/runs/train/exp*```.

## Part 2 Create GPU training clusters and prepare training on Azure Machine Learning

In order to train yolov5 on Azure GPU training clusters, you need to also create datasets that can be accessed by the clusters during training. The first two steps are intended to create data storage and upload the molecule dataset to cloud.

First of all, go to Azure Machine Learning portal and sign in with UW account.Then choose the resource group for this class (named as ```rg-amlclass-<your-uw-id>```), and you will find a Azure Machine Learning Studio resource named as ```amlclass<your-uwid>```: 
<img src="./images/navigate_to_amls_step1.png" style="height: 90%; width: 90%;"/>

Click that resources, and in the following page click ```Launch studio```, you will be navigated to Azure Machine Learning Studio Home page. Also note that you can find the your storage account under this page, which you will be used for creating datastores.
<img src="./images/navigate_to_amls_step2.png" style="height: 90%; width: 90%;"/>

### Step A. Create a DataStore
Go to your Azure Machine Learning Studio Home page and selete ```Create new```, in the scrolled list, select ```Datastore```
<img src="./images/create_datastore_step1.png" style="height: 90%; width: 90%;"/>

In the prompted file, fill in all the fields other than ```SAS token``` as this:
<img src="./images/create_datastore_step2.png" style="height: 90%; width: 90%;"/>

For ```SAS token```, go back to your resource group page, select and click the corresponding storage account used by Machine Learning studio:
<img src="./images/create_datastore_step3.png" style="height: 90%; width: 90%;"/>

In the left side bar of the following page, search ```sas```, and then choose ```Shared access signature```
<img src="./images/create_datastore_step4.png" style="height: 30%; width: 30%;"/>

Once the right side is prompted, select all of the ```Allowed resource types```, double check the ```start and epiry data/time``` to make sure it cover the range of this tutorial. Then click ```Generate SAS and connection string``` 
<img src="./images/create_datastore_step5.png" style="height: 90%; width: 90%;"/>

The SAS token should be sucessfully generated at the end of this page:
<img src="./images/create_datastore_step6.png" style="height: 90%; width: 90%;"/>

Copy the ```SAS token``` and paste it back to previous datastore creation, and click ```Create```. Then you will successfully created a datastore:
<img src="./images/create_datastore_step7.png" style="height: 90%; width: 90%;"/>

### Step B. Create a DataSets of molecule images

Open your terminal and navigate to the folder you created (```MSE544_yolo_training```) in part 1. Tar the whole dataset in order to keep the data structure during upload.
```
tar -cvf molecule_images.tar ./molecule_images
```

Navigate back to your Home page of your Azure Machine Learning studio, choose ```Datasets``` from left side bar and then click ```+ Create dataset``` at the right window. Choose ```From local files``` in the scoll-up list.
<img src="./images/create_dataset_step1.png" style="height: 90%; width: 90%;"/>

Fill in the basic information in the prompted window and go ```Next```
<img src="./images/create_dataset_step2.png" style="height: 90%; width: 90%;"/>

In the following page, choose the one datastore you created in last step within ```previous created datastore``` and then click ```Select datastore```
<img src="./images/create_dataset_step3.png" style="height: 90%; width: 90%;"/>

Then click ```Browse```, in the scrolled list, choose ```Upload files```:
<img src="./images/create_dataset_step4.png" style="height: 50%; width: 50%;"/>

and select the tar file you made earlier ```molecule_image.tar```:
<img src="./images/create_dataset_step5.png" style="height: 50%; width: 50%;"/>

Click ```Next``` to initialize the uploading:
<img src="./images/create_dataset_step6.png" style="height: 90%; width: 90%;"/>

Once the uploading is finished, click ```Create``` in the summary page:
<img src="./images/create_dataset_step7.png" style="height: 90%; width: 90%;"/>

And a dataset is successfully created.
<img src="./images/create_dataset_step8.png" style="height: 90%; width: 90%;"/>

### Step C. Create a GPU Training Cluster
Navigate back to your home page of your Azure Machine Learning studio, and this time use ```Create new``` to create a ```Training cluster```
And a dataset is successfully created.
<img src="./images/create_GPU_cluster_step1.png" style="height: 90%; width: 90%;"/>

In the prompted window, choose options as this screenshot, and then click ```Next``` 
<img src="./images/create_GPU_cluster_step2.png" style="height: 90%; width: 90%;"/>

In the following page, name the GPU cluster as ```GPU-<your-uw-id>```, and set ```Idle seconds before scale down``` as ```120``` seconds. The other options remains as defaults. Then click ```Create```:
<img src="./images/create_GPU_cluster_step3.png" style="height: 90%; width: 90%;"/>

Then you GPU cluster will be succussfully created:
<img src="./images/create_GPU_cluster_step4.png" style="height: 90%; width: 90%;"/>

You can click the into your GPU cluster to obtain the configuration information that will be used for submitting the jobs:
Then you GPU cluster will be succussfully created:
<img src="./images/create_GPU_cluster_step5.png" style="height: 90%; width: 90%;"/>


## Part 3 Train YoloV5 on Azure Machine Learning


### Step A. Set up configurations for training environment

Switch back to the notebook from part 1, and add a new cell and import helper functions from ```azureml.core```
```
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
```

Define the python environment on GPU cluster:
```
yolov5_env = Environment(name="yolov5_env")

yolov5_env.docker.base_image  = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04"

conda_dep = CondaDependencies()
conda_dep.add_conda_package('python=3.8')

# install all the yolov5 requirement at the image build time
with open('./yolov5/requirements.txt', 'r') as f:
    line = f.readline()
    
    while line != '':        
        if line.startswith('#') or len(line.split()) == 0:
            line = f.readline()
            continue
        
        conda_dep.add_pip_package(line.split()[0])
        line = f.readline()

yolov5_env.python.conda_dependencies=conda_dep
```

you can check the details of the enviroment you defined using
```
yolov5_env.get_image_details
```

And confirm that the python version is 3.8 or later from the output:
```
...
            "dependencies": [
                "python=3.8",
                {
                    "pip": [
                        "azureml-defaults",
                        "matplotlib>=3.2.2",
                        "numpy>=1.18.5",
                        "opencv-python>=4.1.2",
                        "Pillow",
                        "PyYAML>=5.3.1",
                        "scipy>=1.4.1",
                        "torch>=1.7.0",
                        "torchvision>=0.8.1",
                        "tqdm>=4.41.0",
                        "tensorboard>=2.4.1",
                        "seaborn>=0.11.0",
                        "pandas",
                        "thop",
                        "pycocotools>=2.0"
                    ]
                }
            ],
...
```

### Step B. Create a training script
Open your terminal or use your GUI interface on your computer to navigate to the folder you created (```MSE544_yolo_training```) in part 1. Create a new folder (named by ```deploy_yolo_training```), which will be upload to Azure and used as the working directory.
```
mkdir deploy_yolo_training
```  

Go into the new directory, and create a python training script named as ```training_on_aml.py```

Open the file and create the script by the following steps:

- Connect to the datastore and download dataset
    - import necessary packages
    ```
    from azureml.core import Workspace, Dataset, Run
    import os, tempfile, tarfile
    ```
    - Make a temporary directory and mount molecule image dataset
    ```
    mounted_path = tempfile.mkdtemp()
    print('Tmporary directory made at' + mounted_path)

    # locate the molecule_images dataset
    ws = Run.get_context().experiment.workspace
    dataset = Dataset.get_by_name(ws, name='molecule_images_yolov5')

    # mount data 
    mount_context = dataset.mount(mounted_path)
    mount_context.start()
    
    print("molecule_images dataset mounting done")
    ```
    - Untar files to the working directory
    ```
    # untar all files under this directory, 
    for file in os.listdir(mounted_path):
        if file.endswith('.tar'):
            tar = tarfile.open(os.path.join(mounted_path, file))
            tar.extractall()
            tar.close()
    ```
- Set up yolov5 environment, very similar as part 1
    ```
    # this is needed for container
    os.system('apt-get install -y python3-opencv')
    
    os.system('git clone https://github.com/ultralytics/yolov5')
    
    # since it's already in a container, no need to make new conda environment
    os.chdir('./yolov5')
    
    # no need to install the requirements again
    # os.system('pip install -r requirements.txt')

    # check GPU
    import torch
    print(f"yolov5 enviroment setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
    ```
-   Add training command and start training
    ```
    # make sure you have --save-dir to ../outputs/ in order to save all results after running
    os.system('python train.py --img 640 --batch 16 --epochs 100 --data ../molecule_images/molecule_detection_yolov5.yaml --weights yolov5s.pt --save-dir ../outputs/')
    ```

### Step C. Submit the job and do the yolov5 training on cloud
Now swith back to the notebook again, and set up experiment:
```
subscription_id = '<your_subscription_id>'
resource_group  = '<your_resoure_group>'
workspace_name  = '<your_workspace_name>'
ws = Workspace(subscription_id, resource_group, workspace_name)

experiment = Experiment(workspace=ws, name='molecule_detection_yolo_training')
```

Then create script run configurations as:
```
# Overall configuration for the script to be run on the compute cluster
config = ScriptRunConfig(source_directory='./deploy_yolo_training/',   ## folder in which the script is located
                         script='training_on_aml.py',       ## script name
                         compute_target='<your-gpu-cluster-name>',
                         environment=yolov5_env)   
```

All the field within each ```<>``` can be found at the end of Part 2 Step C and they need to be replaced with your own values before proceeding to next cell. 

Check the running directory of your notebook by 
```
%pwd
```
and if not in folder ```MSE544_yolo_training```, switch to it by
```
%cd <path-to-MSE544_yolo_training>
```

Submit the job to the GPU cluster on Azure by execute the following in your notebook:
```
run = experiment.submit(config)
aml_url = run.get_portal_url()
print(aml_url)
```

If the training job is successfully deployed, a url will be printed as output. Click the url will navigate you to the experiment you submitted on the Azure Machine Learning studio.
 <img src="./images/experiment_url.png" style="height: 90%; width: 90%;"/>


### Step D. Check the running logs and download the weights

### Step E. Inference using YoloV5

For this step, you can either download a pre-trained weights from this git repository(FIXME:comming soon), or wait until the end when you obtain a weights form cloud training.

To run test, simply use the following command under yolov5 directory:
```
!python test.py --weights test_weights.pt --data ../molecule_images/test/images --iou 0.80
```

The results of your inference is will be located at ```yolov5/runs/test/exp*```

## Reference





