# Sartorious
Welcome  to the Sartorius project. Project Members: Hem Regmi, Yuqi Wu
# Unet Model
- The Unet model is based on the architecture from: https://arxiv.org/pdf/1505.04597.pdf
- The image size for the project is (256, 256, 3), and hence input pattern is needed to modified.
- To Train model
  - Create the project in pycharm with following requirements.txt under Unet folder
  - download the data from the following dropbox link shared here:https://www.dropbox.com/sh/2uptg7fwntzze38/AAA8iVNC5I3_YqS_6OyGMmnwa?dl=0
  - run the main.py file to train the network which splits the training data into training and validation and train for 100 epochs
  - training time is ~ 7-8 hrs in GPU based machine 
  - It requires the Tensorflow API and is tested on python 3.7
  - The expected output are IoU and DSC scores on validation images over multiple epochs
# Detectron 2 Model
- This model zoo requires linux based system. Unfortunately, it doesn't run in the Windows based machine
- Make sure to create project and install requirments based on requirements.txt
- This model is running based on PyTorch API and tested on python 3.7
- To Train Model:
    - create the project in pycharm or any other IDE and install all the dependencies
    - download the data and code from: https://www.dropbox.com/sh/2uptg7fwntzze38/AAA8iVNC5I3_YqS_6OyGMmnwa?dl=0
    - run the main.py to train the model
    - run evaluate.py to evaluate and print the output images
    - run plotdata.py to generate the instance segmented and cell detected images for multiple images
 # To do list
 - implement the tensorboard to log more data from training
 - use the unlabelled data also to improve model and accuracy
 - try Transformers for segmentation

