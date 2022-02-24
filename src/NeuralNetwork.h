#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Algebra.h"
#include <string.h>
#include <stdint.h>

#ifdef BMP
#include "bmp.h"
#endif

#define MAX_LENGHT 320
#define DATA_SIZE (28*28)
#define LABEL_SIZE (10)
#define BLACK 255
#define WHITE 00

#define DEF_ 0
#define SIGMOID 1
#define RELU 2
#define TANH 3
#define SOFTMAX 4

typedef struct _layer 
{
   vector activations;
   vector z;
   matrix weights; /*from previous layer to current one*/
   vector biases;  /*from previous layer to current one*/
   struct _layer* next;
   struct _layer* prev;
   int function;
   int lenght;
} layer;

typedef struct 
{
   int n_layers;
   layer* output;
   layer* input;
} neural_network;

typedef struct 
{
   vector data;
   vector label;
} labeled_data;

typedef struct
{
   int n_batches;
   int batch_size;
   labeled_data** data;
} training_data;

/*network creation functions*/
neural_network create_neural_network (int number_layers, int net_shape[], int act_func);
void delete_neural_network (neural_network* n);
void add_layer (neural_network* n, int dim, int act_func);
layer* create_layer (int dim, int act_func);
void delete_layer (layer* l);
void refresh_layer(layer* l);
void tr_refresh_layer(layer* l, vector* z);
vector prediction (neural_network n, vector input_data);
void randomize_network (neural_network *n);
void zero_neural_network (neural_network* n);

/*training functions*/
void train (neural_network* n, training_data data, int epochs, float learning_rate, float momentum, training_data test, int testing_interval);
void training_session (neural_network* n, training_data data, float learning_rate, float momentum);
neural_network stochastic_gradient_descent (neural_network* n, labeled_data* batch, int batch_size, float learning_rate, neural_network prec, float momentum);
neural_network create_neural_network_copy (neural_network n);
void scalar_neural_network_product (neural_network* n, float s);
void neural_network_subtraction (neural_network sour, neural_network* dest);
void total_gradient (neural_network n, neural_network* temp, labeled_data* batch, int batch_size);
void backpropagation (layer* temp, layer* l, vector diff);
void add_sample_gradient (matrix* wg, vector* bg, vector* ag, layer* l, vector difference);
void bias_sample_gradient (vector *z, vector difference, int act_func);
void weight_sample_gradient (matrix* g, vector bias_g, vector prev_act);
void activation_sample_gradient (vector* g, vector bias_g, matrix weights);

/*data management functions*/
training_data load_training_data (FILE* data_file, FILE* labels_file, int batch_size);
training_data create_training_data (int n_batches, int batch_size);
void read_mnist_header (FILE* data_file, FILE* labels_file, int32_t* n_samples, int32_t* rows, int32_t* columns);
void delete_training_data (training_data* d);
void shuffle_training_data (training_data* d);
labeled_data create_labeled_data (int data_size, int label_size);
void delete_labeled_data (labeled_data* d);
void extract_labeled_data_from_mnist (labeled_data* d, FILE* data_file, FILE* labels_file, int32_t rows, int32_t columns);
#ifdef BMP
BITMAP read_bitmap_from_mnist (FILE* fp, int n);
void extract_bitmap_from_mnist (FILE* fp, BITMAP img);
#endif
vector read_vector_from_mnist (FILE* fp, int n);
void extract_vector_from_mnist (FILE* fp, vector* img);

float test_performances (neural_network n, training_data d);

/*load and store functions*/
void load_neural_network (FILE* fp, neural_network* n);
void store_neural_network (neural_network n, FILE* fp);
void store_neural_network_bin (FILE* fp, neural_network n);
void store_layer_bin (FILE* fp, layer l);
neural_network load_neural_network_bin (FILE* fp);
layer* load_layer_bin (FILE* fp);


/*activation functions*/
float sigmoid (float x);
float sigmoid_derivative (float x);
float relu (float x);
float relu_derivative (float x);
float tan_h (float x);
float tanh_derivative (float x);
void softmax (vector* v);
void softmax_derivative (vector *v);

u_int32_t be32_toh (u_int32_t n);

