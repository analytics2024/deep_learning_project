# image-captioning
This code works on Keras 2.15.0 version

Steps to run this code
1) Download the flickr8k dataset from https://www.kaggle.com/datasets/adityajn105/flickr8k
2) Download the Glove embeddings from https://www.kaggle.com/datasets/anmolkumar/glove-embeddings
3) Firstly run the feature_extraction.ipynb to create a feature.pkl file. This will run a CNN model on your images and create the feature.pkl file.
4) Then run Caption_preprocessing.ipynb file. This will do the preprocessing iof your captions
5) Finally run the model training and testing notebook.

