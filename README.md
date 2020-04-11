# Anomaly Detection Using Transfer Learning

### Model: 
Pre-trained VGG16 convolution layers + new classification layers added.
Portion of the convolution layers are frozen.
### Dataset Augmentation: 
Color jitter, Random Vertical Flip/ Horizontal Flip/ Rotation.
You can try testing the model by training it after changing these parameters.

### Data Preprocessing
For data preprocessing please follow instructions in the preprocessing folder.

### Environment and installation instructions
1. setup a virtual environment named anomaly
`conda create -n anomaly python=3.6`
2. Activate the environment: `conda activate anomaly`
3. Install PyTorch: `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
4. Install python requirements `pip install -r requirements.txt`
5. Make sure the directory `./preprocessing/data` exists and contains the `/normal` and `/anomaly` folders with images.
6. Open the file `config.py` and setup the parameters for training and testing.
7. Split the data set into train/validation/test folders: `python3 split.py`
8. Train the model: `python3 run_training.py`
This command downloads vgg16 model, modifies it, creates the dataloader, trains the model, and stores the trained model and history of training in folder `/pretrained`.
Loss history curve is displayed (remember to close window to continue).
9. Test the model by running `python3 test.py`..


 
