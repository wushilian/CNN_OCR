# CNN_OCR
use CNN to recognize charset

# How to train a model
first you should prepare your image in 'train' dir,and the image name should like '0_A.jpg','1_B.jpg'

then you should prepare your image in 'test' dir,image name should like train image

after prepared image,you should edit the config.py to change image width and other things

at last,run the run.py

# How use trained model to label origin image
1.move trained model's weight file to pretrained dir

2.changeg config.py pretrained_path

3.use run.py's function of label_img() to label origin image
