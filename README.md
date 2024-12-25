# gpt_state_generation

This code for the GPT model has been written with the help of the following GitHub page: [Infatoshi/fcc-intro-to-llms](https://github.com/Infatoshi/fcc-intro-to-llms).

# A schematic representation of training data preparation and the architecture of the decoder-only transformer.

The left-hand side figure represents the schematic representation of the discretization of molecular dynamics simulation (MDs) trajectory achieved through the identification of collective variables (CVs) and K-means clustering. A total of $n_s = 10$ segments are randomly selected from the discretized trajectory to train an equal number of independent generative pre-trained transformer (GPT) models. Each trained model generates subsequent sequences starting from where the respective segment ended, using a few sequences as prompts. The right-hand side figure depicts the various layers of a decoder-only transformer. The model architecture consists of input embedding, positional embeddings, and multiple blocks of masked multi-head attention, normalization, and feed-forward layers.

![](transformer_schematic.png)

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


