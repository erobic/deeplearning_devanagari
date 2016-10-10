## Deep Learning model to recognize handwritten devanagari characters

### Setup:

#### Create these directories:

* **data**
* **data/train:** Each sub-directory in this directory should contain images for training. Sample sub-directory: character_10_yna. Download: https://drive.google.com/folderview?id=0B1fb_oJIyNc5SC1rTnZTdU9Sdmc&usp=sharing
* **data/test:** Each sub-directory in this directory should contain test images. Sample sub-directory: character_10_yna. Download: https://drive.google.com/folderview?id=0B1fb_oJIyNc5SC1rTnZTdU9Sdmc&usp=sharing
* **data/tfrecords/train:** step1_images_to_tfrecords.py will write .tfrecords files for training data
* **data/tfrecords/test:** step1_images_to_tfrecords.py will write .tfrecords files for test data

