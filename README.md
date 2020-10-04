# Keras-RetinaNet

This Repository conatains Implementation of RetinaNet from Scratch Using TensorFlow/Keras. The Backbone Used is ResNet50 which is built-in
in TensorFlow/Keras. 
You can train the Model on Your Custom Dataset as well as you can train it on official COCO dataset.

In 'dataset.py' file set data_dir=None to train on full COCO dataset, or leave it as it is to train on smaller dataset (500 images) and you can 
prepare you custom dataset as well.

Then use 'train.py' file to train the model.

After Training you can use 'load_model.py' file to Load the checkpoint weights, or you can download previously trained weights and use it and 
Visualize the Detected Objects in the images.
