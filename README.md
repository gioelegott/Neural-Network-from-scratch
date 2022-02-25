#Neural Network form Scratch

The aim of this project is to create a Neural Network library in C without using any external libraries.
The idea and structure behind this project comes from the 3Blue1Brown youtube series **** and the data used for training and testing comes from the mnist database ***

The library is divided in 2 parts:
- algebra.h: defines the structs matrix and vector (of float) and contains:
   * memory management functions
   * I/O functions
   * algebric operations
  This structs and functions are used in the second part of the library
- NeuralNetwork.h: 
   - structs defined:
     * layer: layer of a neural network made of neurons and their connections (weights and biases) with the previous layer's neurons. All these entities are represented with matrices and vectors.
     * neural network: list of layers
     * labeled data: struct that contains one image and its label (for more reference visit ****)
     * training data: bidimensional array of labelled data
   - functions:
     * neural network creation functions
     * training functions
     * memory management functions
     * load and store functions

To use this library follow these steps:

1 Creating neural network:

Define an array of integers that contains the number of neurons for each layer in order (the first and the last number must always be your input and output size) and pass it to the create_neural_network function with the name of the activation function you want to use (SIGMOID, RELU, TANH or SOFTMAX). If you need to add other layers in tail you can use the add_layer function.
After the creation of the.

2 Loading training data:

You can load the training data from mnist database using the load_training_data function specifying the number of batches into wich you want to divide the data. This function can be used both for training and testing data.

3 Training:

Before training randomize the values of your network using the randomize_network function.
The most important function for training is train() and its parameters are:
 - the pointer to the neural network you want to train
 - the training data
 - the number of epochs
 - the learning rate
 - the momentum
 - the test data to determine the accuracy every X epochs
 - the number X

4 Using the trained neural network

aisijdiaia

Examples of its use are MnistTraining.c and MnistClassification.c and the results of different sessions of training can be found int he saves folder.

This library is only a proof of concept and it should not be used for any reason except the testing of its own functioning.





