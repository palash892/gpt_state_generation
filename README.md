# gpt_state_generation

This code for the GPT model has been written with the help of the following GitHub page: [Infatoshi/fcc-intro-to-llms](https://github.com/Infatoshi/fcc-intro-to-llms).

# Code Requirements

Ensure you have the following Python packages installed to run the code:

1. [numpy](https://numpy.org/)
2. [tensorflow](https://www.tensorflow.org/)
3. [PyTorch](https://pytorch.org/)

# Package version
1. Python (3.11.5)
2. Numpy  (1.26.4)
3. PyTorch (2.2.0)
4. Tensorflow (2.15.0)

# How to run the code

There are mainly two folders, `autoencoder` and `gpt_model`, which contain their corresponding code for training the model. For `gpt_model`, the data for three-state models is provided. This folder contains the data for training, validation, and generation. After completion of the training by running the Python scripts `train_gpt_model.py`, one can generate as many sequences as possible by running the Python script `inference.py`.


