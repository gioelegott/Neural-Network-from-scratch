# Neural Network form Scratch

The aim of this project is to create a Neural Network library in C without using any external libraries.
The idea and structure behind this project comes from the 3Blue1Brown youtube series https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi and the data used for training and testing comes from the mnist database http://yann.lecun.com/exdb/mnist/.

The library is divided in 2 parts:
- algebra.h: defines the structs matrix and vector (of float) and contains:
   * memory management functions
   * I/O functions
   * algebric operations
  These structs and functions are used in the second part of the library
- NeuralNetwork.h: 
   - structs defined:
     * layer: layer of a neural network made of neurons and their connections (weights and biases) with the previous layer's neurons. Weights, biases and neurons are represented with matrices and vectors.
     * neural network: list of layers
     * labeled data: struct that contains one image and its label (for more reference visit http://yann.lecun.com/exdb/mnist/)
     * training data: bidimensional array of labelled data
   - functions:
     * neural network creation functions
     * training functions
     * memory management functions
     * load and store functions

To use this library follow these steps:

1 Creata a neural network:

Define an array of integers that contains the number of neurons for each layer in order (the first and the last number must always be your input and output size) and pass it to the create_neural_network function with the name of the activation function you want to use (SIGMOID, RELU, TANH or SOFTMAX). If you need to add other layers on the tail of the list you can use the add_layer function.

2 Load the training data:

You can read the training data obtained from mnist database using the load_training_data function specifying the number of batches into wich you want to divide the data. This function can be used both for training and testing data.

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

4 Store the trained network

After the training session save the trained neural network into a file (.nn or .nnb) using the store_neural_network or store_neural_network_bin functions. Before ending your program remember to deallocate the allocated memory using delete_neural_network and delete_training_data functions.

5 Use the trained neural network

You can use a neural network by loading it from a file (examples can be found in the saves folder) and then using it to make predictions with the function predict or test its performances with the function test_performances.

Examples of this library's usage are MnistTraining.c and MnistClassification.c and the results of different sessions of training can be found in the saves folder.

This library is only a proof of concept and it should not be used for any reason except the testing of its own functioning.





