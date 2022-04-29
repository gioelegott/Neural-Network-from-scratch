#include "NeuralNetwork.h"


/*********  NEURAL FUNCTIONS  ************/

neural_network create_neural_network (int number_layers, int net_shape[], int act_func)
{
   /*returns a neural network with number_layers layers; their size is determined by net_shape*/

   int i, j, k;
   neural_network n;

   if (number_layers < 1)
   {
      fprintf(stderr, "you need more than 2 layers to create a network\n");
      return n;
   }

   /*creating first layer*/
   layer *l1 = create_layer(net_shape[0], act_func);
   n.input = l1;


   /*creating second layer*/
   layer *l2 = create_layer(net_shape[1], act_func);
   n.output = l2;
   
   /*connecting first and second layer*/
   n.input->next = l2;
   n.output->prev = l1;
   n.n_layers = 2;

   /*creating first and second layer's weights and biases*/
   l1->biases = create_vector(0);
   l1->weights = create_matrix(0,0);
   
   l2->biases = create_vector(l2->activations.lenght);
   l2->weights = create_matrix(l2->activations.lenght, l2->prev->activations.lenght);

   /*creating the other layers by adding them to the last layer*/
   for (i = 2; i < number_layers; i++)
   {
      add_layer(&n, net_shape[i], act_func);      
   }

   return n;
}

void delete_neural_network (neural_network* n)
{
   /*deletes n by deallocating its memory portion*/

   layer* l = n->input;
   /*deleting layers*/
   while ((l = l->next) != NULL)
   {
      delete_layer(l->prev);
   }

   delete_layer(n->output);

   n->input = NULL;
   n->output = NULL;
   n->n_layers = 0;
   return;
}

void add_layer (neural_network* n, int dim, int act_func)
{
   /*adds a new layer at the end of n*/

   /*new layer pointers*/
   layer* l = create_layer(dim, act_func);
   l->prev = n->output;
   l->next = NULL;

   /*new output layer for n*/
   n->output->next = l;
   n->output = l;   
   n->n_layers += 1;

   /*creating new layer's weights and biases*/
   l->biases = create_vector(l->lenght);
   l->weights = create_matrix(l->lenght, l->prev->lenght);
   
   return;
}

layer* create_layer (int dim, int act_func)
{
   /*allocates memory for the layer and activation and z vectors (not weights nor biases)*/
   layer* l = (layer*)malloc(sizeof(layer));
   l->prev = NULL;
   l->next = NULL;
   l->activations = create_vector(dim);
   l->z = create_vector(dim);
   l->function= act_func;
   l->lenght = dim;

   return l;
}

void delete_layer (layer* l)
{
   /*deletes l by deallocating its memory portion*/
   l->prev = NULL;
   l->next = NULL;
   l->function = DEF_;

   delete_vector(&(l->activations));
   delete_vector(&(l->z));
   delete_vector(&(l->biases));
   delete_matrix(&(l->weights));
   free(l);

   return;
}

void refresh_layer(layer* l)
{
   /*updates l activations (and z)*/

   /*weighted sum of prevous layer's actiations*/
   matrix_vector_product(l->weights, l->prev->activations, &(l->z));

   /*adding biases*/
   vector_sum(l->biases, &(l->z));

   /*activation function*/
   copy_vector (l->z, &(l->activations));

   switch(l->function)
   {
      case SIGMOID:
      vector_function(&(l->activations), sigmoid);
      break;
      case RELU:
      vector_function(&(l->activations), relu);
      break;
      case TANH:
      vector_function(&(l->activations), tan_h);
      break;
      case SOFTMAX:
      softmax(&(l->activations));
      break;
      case DEF_:
      fprintf(stderr, "accessing to temporary network function\n");
      break;

      fprintf(stderr, "function %d not available\n", l->function);
   }
   /*if (!strcmp(l->function, "sigmoid"))
      vector_function(&(l->activations), sigmoid);
   else if (!strcmp(l->function, "relu"))
      vector_function(&(l->activations), relu);
   else if (!strcmp(l->function, "tanh"))
      vector_function(&(l->activations), tanhf);
   else if (!strcmp(l->function, "softmax"))
      softmax(&(l->activations));
   else if (!strcmp(l->function, ""))
      fprintf(stderr, "accessing to temporary network function\n");
   else
      fprintf(stderr, "function %s not available\n", l->function);*/
   
   return;
}

vector prediction (neural_network n, vector input_data)
{
   /*returns a vector whose highest value represents the predicted classification for input_data*/

   layer* l;
   if (input_data.lenght != n.input->activations.lenght)
   {
      fprintf(stderr, "wrong input data. Data: %d Layer: %d\n", input_data.lenght, n.input->activations.lenght);
      return n.output->activations;
   }

   /*inserting input_data as first layer's activations*/
   copy_vector(input_data, &(n.input->activations));

   /*updating all following layers*/
   l = n.input;
   while ((l = l->next) != NULL)
   {
      refresh_layer(l);
   }

   return n.output->activations;
}

void randomize_network (neural_network *n)
{
   /*sets all weights and biases of n with a random number between -1 and 1*/

   layer* l = n->input;
   /*for each layer randomizing its weight matrix and biases vector*/
   while((l = l->next) != NULL)
   {
      randomize_vector(&(l->biases), -0.5, 0.5);
      randomize_matrix(&(l->weights), -0.5, 0.5);
   }
   return;
}


/*********  TRAINING FUNCTIONS  ************/

void train (neural_network* n, training_data data, int epochs, float learning_rate, float momentum, training_data test, int testing_interval)
{
   /*trains n by learning from the training data wich is elaborated epochs times*/

   int i, j, k;
   int num_files;
   num_files = data.n_batches * data.batch_size;
   float dev;


   /*randomizig data order and grouping and training n for each epoch */
   for (i = 0; i < epochs; i++)
   {
      fprintf(stderr, "EPOCH %d of %d   ", i+1, epochs);

      clock_t begin = clock();

      shuffle_training_data(&data);

      /*training n with the current order of data*/
      training_session (n, data, learning_rate, momentum);//((float)i/5 +1));

      clock_t end = clock();
      double time_spent = (double)(end - begin)/CLOCKS_PER_SEC;

      int minutes = (int)time_spent/60;
      float seconds = (float)time_spent - (minutes*60);
      fprintf(stderr, "%d' %g''\n", minutes, seconds);

      if (i%testing_interval == 0)
      {
         print_vector(n->input->next->biases);
         //print_matrix(n->input->next->weights);
         dev = test_performances(*n, test);

         fprintf(stderr, "Dev: %g\n\n", dev);
      }
      
   }
   return;

}

void training_session (neural_network* n, training_data data, float learning_rate, float momentum)
{
   /*changes n weights and biases according to the data set with a velocity coeff*/

   int i, m;
   neural_network temp = create_neural_network_copy(*n);
   zero_neural_network(&temp);

   /*taking a step to optimize the cost function for each batch*/

   for (i = 0; i < data.n_batches; i++)
   {
      m = data.n_batches/20;
      if (i%m == 0)
         fprintf(stderr, "*");

      temp = stochastic_gradient_descent(n, data.data[i], data.batch_size, learning_rate, temp, momentum);
      //printf("\n%d/%d batches done!\n", i+1, data.n_batches);
   }

   fprintf(stderr, "\n");
   delete_neural_network(&temp);

   return;
}

neural_network stochastic_gradient_descent (neural_network* n, labeled_data* batch, int batch_size, float learning_rate, neural_network prec, float momentum)
{
   /*optimizes the cost/loss function accordingly to the current batch of data with a velocity coeff*/

   /*creating  a neural network to store all the backpropagation results*/
   neural_network temp = create_neural_network_copy (*n);
   zero_neural_network (&temp);

   /*calculating the sum of all samples' backpropagation results*/
   total_gradient (*n, &temp, batch, batch_size);


   /*averaging the results of backropagtion by dividing it for the batch size and then multipling by the coefficient*/
   scalar_neural_network_product (&temp, (float)1.0/(float)batch_size * learning_rate);

   if (momentum != 0.0)
   {
      scalar_neural_network_product (&prec, -1 * momentum);
      neural_network_subtraction (prec, &temp);
   }

   /*subtracting the result stored in the temporary network to the actual neural network*/
   neural_network_subtraction (temp, n);

   /*deleteng the temporary neural network*/
   delete_neural_network (&prec);
   return temp;
}

neural_network create_neural_network_copy (neural_network n)
{
   int j;
   int net_shape[n.n_layers];
   layer *l = n.input;

   for (j = 0; j < n.n_layers; j++)
   {
      net_shape[j] = l->lenght;
      l = l->next;
   }
   neural_network temp = create_neural_network(n.n_layers, net_shape, DEF_);
   
   return temp;
}

void scalar_neural_network_product (neural_network* n, float s)
{

   layer* l = n->input;
   while ((l = l->next) != NULL)
   {
      scalar_matrix_product (&(l->weights), s);
      scalar_vector_product (&(l->biases), s);
   }

   return;
}

void neural_network_subtraction (neural_network sour, neural_network* dest)
{
   layer* l_sour = sour.input;
   layer* l_dest = dest->input;
   while ((l_sour = l_sour->next) != NULL)
   {  
      l_dest = l_dest->next;
      matrix_subtraction (l_sour->weights, &(l_dest->weights));
      vector_subtraction (l_sour->biases, &(l_dest->biases));
   }
   return;
}

void total_gradient (neural_network n, neural_network* temp, labeled_data* batch, int batch_size)
{
   /*calculates the sum of all samples' backpropagation results and stores them in the temporary neural network*/

   int i;
   layer* out = n.output;
   vector diff;

   /*creating a structure to contain the current sample*/

   layer* p_out_temp = temp->output;

   /*computing the prediction of each sample and adding the backpropagation results to the temporary network*/
   for (i = 0; i < batch_size; i++)
   {
      //printf("*");

      /*calculating the current sample prediction*/
      diff = prediction (n, batch[i].data);


      /*calculating the difference between the right prediction (label) and the prediction given by network*/
      vector_subtraction (batch[i].label, &diff);

      /*adding the adjustments needed to have a better performance in the temporary network*/
      //printf("qui\n");

      backpropagation (p_out_temp, out, diff);
   }

   return;
}

void backpropagation (layer* temp, layer* l, vector diff)
{
   /*calculates recursivly the adjustements needed to better perform with the current data sample (diff) and adds them to the current temporary layer*/

   /*computing the gradient by adding the current sample influence*/
   add_sample_gradient (&(temp->weights), &(temp->biases), &(temp->prev->activations), l, diff);

   /*making sure that backpropagation is possible (previous layer must not be the input layer)*/
   if (l->prev->prev != NULL)
   {
      /*recursive call propagting backwards in the network and using the activation gradient as diff*/
      backpropagation (temp->prev, l->prev, temp->prev->activations);
   }

   
   return;
}



void add_sample_gradient (matrix* wg, vector* bg, vector* ag, layer* l, vector difference)
{
   /*calculates weights, biases and activations sample gradient and adds them to the corrisponding data structure*/

   vector temp_v1 = create_copy_vector (l->z);
   vector temp_v2 = create_vector (ag->lenght);
   matrix temp_m = create_matrix (wg->rows, wg->columns);

   /*baiases gradient*/
   bias_sample_gradient (&temp_v1, difference, l->function);
   vector_sum (temp_v1, bg);

   /*weight gradient*/
   weight_sample_gradient (&temp_m, temp_v1, l->prev->activations);
   matrix_sum (temp_m, wg);

   /*activation gradient*/ //vector_matrix_product
   activation_sample_gradient(&temp_v2, temp_v1, l->weights);
   vector_sum (temp_v2, ag);

   delete_matrix(&temp_m);
   delete_vector(&temp_v1);
   delete_vector(&temp_v2);

   return;
}

void bias_sample_gradient (vector *z, vector difference, int act_func)
{

   /*calculates biases gradient: sigm'(z) * 2 * (a-y) */
   if (z->lenght != difference.lenght)
   {
      fprintf(stderr, "bias gradient cannot be calculated\n");
      return;
   }

   /*calculating the activation function derivative of z (weighted sum)*/

   switch(act_func)
   {
      case SIGMOID:
      vector_function(z, sigmoid_derivative);
      break;
      case RELU:
      vector_function(z, relu_derivative);
      break;
      case TANH:
      vector_function(z, tanh_derivative);
      break;
      case SOFTMAX:
      softmax_derivative(z);
      break;
      case DEF_:
      fprintf(stderr, "accessing to temporary network function\n");
      break;

      fprintf(stderr, "function %d not available\n", act_func);
   }
   /*multiplying by 2 times the difference*/
   int i;
   for (i = 0; i < difference.lenght; i++)
   {
      z->entry[i] *= 2*difference.entry[i];
   }

   return;
}

void weight_sample_gradient (matrix* g, vector bias_g, vector prev_act)
{
   /*calculates weights gradient: sigm'(z) * 2 * (a-y) * a_prev = bias_g * a_prev */
   if (g->rows != bias_g.lenght || g->columns != prev_act.lenght)
   {
      fprintf(stderr, "weight gradient cannot be calculated\n");
      return;
   }

   /*multiplying vectors bias_g and prev_act considering them as matrices (vertical * orizontal)*/
   int i, j;
   for (i = 0; i < g->rows; i++)
   {
      for (j = 0; j < g->columns; j++)
      {
         g->entry[i][j] = bias_g.entry[i] * prev_act.entry[j];
      }
   }
}

void activation_sample_gradient (vector* g, vector bias_g, matrix weights)
{
   /*calculates actvation gradient: sum(sigm(z) * 2 * (a-y) * weight) = sum(bias_g * weights)*/

   if(bias_g.lenght != weights.rows)
   {
      fprintf(stderr, "activation gradient cannot be calculated\n");
      return;
   }

   /*multiplying vector baias_g (orizontal) by matrix weights*/
   int i, j;
   float p_sum;

   for (j = 0; j < weights.columns; j++)
   {
      p_sum = 0;
      for (i = 0; i < bias_g.lenght; i++)
      {
         p_sum += bias_g.entry[i]*weights.entry[i][j];
      }
      g->entry[j] = p_sum;
   }
   return;
}

void zero_neural_network (neural_network* n)
{
   layer* l = n->input;
   while ((l = l->next) != NULL)
   {
      zero_matrix(&(l->weights));
      zero_vector(&(l->biases));
   }
   return;
}

/*********  DATA FUNCTIONS  ************/

training_data load_training_data (FILE* data_file, FILE* labels_file, int batch_size)
{
   /*creates the training_data structure and loads data from the mnist files*/

   training_data d;
   int j, k;
   int32_t n_samples, rows, columns;

   read_mnist_header (data_file, labels_file, &n_samples, &rows, &columns);

   d = create_training_data(n_samples/batch_size, batch_size);

   for (j = 0; j < d.n_batches; j++)
   {
      for (k = 0; k < d.batch_size; k++)
      {
         d.data[j][k] = create_labeled_data(DATA_SIZE, LABEL_SIZE);
         extract_labeled_data_from_mnist(&(d.data[j][k]), data_file, labels_file, rows, columns);

      }
   }

   return d;
}

training_data create_training_data (int n_batches, int batch_size)
{
   /*allocates memory for training_data*/

   int i;
   training_data d;
   d.n_batches = n_batches;
   d.batch_size = batch_size;

   d.data = (labeled_data**)malloc(sizeof(labeled_data*)*d.n_batches);
   for (i = 0; i < d.n_batches; i++)
   {
      d.data[i] = (labeled_data*)malloc(sizeof(labeled_data)*d.batch_size);
   }

   return d;
}

void read_mnist_header (FILE* data_file, FILE* labels_file, int32_t* n_samples, int32_t* rows, int32_t* columns)
{
   /*reads the mnist data header*/

   /*reading input data's header*/
   int32_t n_labels;
   fseek(data_file, 4, SEEK_SET);
   fread(n_samples, sizeof(*n_samples), 1, data_file);
   *n_samples = (int32_t)be32toh((uint32_t)(*n_samples));

   fread(rows, sizeof(*rows), 1, data_file);
   fread(columns, sizeof(*columns), 1, data_file);
   *rows = (int32_t)be32toh((uint32_t)(*rows));
   *columns = (int32_t)be32toh((uint32_t)(*columns));
 
   /*reading label's header*/
   fseek(labels_file, 4, SEEK_SET);
   fread(&n_labels, sizeof(n_labels), 1, labels_file);
   n_labels = (int32_t)be32toh((uint32_t)(n_labels));
 
   if(*n_samples != n_labels)
   {
      fprintf(stderr, "error in training data, data: %d label: %d\n", *n_samples, n_labels);
      return;
   }

   return;

}

uint32_t be32toh (uint32_t n)
{
   uint32_t c1 = n>>24;
   uint32_t c2 = n<<24;
   uint32_t c3 = n>>8;
   uint32_t c4 = n<<8;

   uint32_t m = c1&0xff | c2&0xff000000 | c3&0xff00 | c4&0xff0000;
   return m;
}

void delete_training_data (training_data* d)
{
   int j, k;
   for (j = 0; j < d->n_batches; j++)
   {
      //printf("j %d ", j);
      for (k = 0; k < d->batch_size; k++)
      {
         //printf("k %d ", k);
         delete_labeled_data(&(d->data[j][k]));
      }
   }

   for (j = 0; j < d->n_batches; j++)
   {
      free(d->data[j]);
   }

   free(d->data);

   d->batch_size = 0;
   d->n_batches = 0;

   return;
}

void shuffle_training_data (training_data* d)
{
   /*shuffles the training data*/
   srand((unsigned int)clock());

   int i;
   labeled_data temp;
   int r1, r2, c1, c2;

   for (i = 0; i < d->n_batches * d->batch_size /2; i++)
   {
      r1 = int_random_number (0, d->n_batches -1);
      r2 = int_random_number (0, d->n_batches -1);

      c1 = int_random_number (0, d->batch_size -1);
      c2 = int_random_number (0, d->batch_size -1);

      temp = d->data[r1][c1];

      d->data[r1][c1] = d->data[r2][c2];

      d->data[r2][c2] = temp;

   }  

   return;
}

labeled_data create_labeled_data (int data_size, int label_size)
{
   /*creates the labeled_data structure*/

   labeled_data d;
   d.data = create_vector(data_size);
   d.label = create_vector(label_size);
   return d;
}

void delete_labeled_data (labeled_data* d)
{
   /*deletes d by deallocating its memory portion*/

   delete_vector(&(d->data));
   delete_vector(&(d->label));

   return;
}

void extract_labeled_data_from_mnist (labeled_data* d, FILE* data_file, FILE* labels_file, int32_t rows, int32_t columns)
{
   unsigned char pixel, label;
   int i, j;

   if (d->data.lenght != rows*columns)
   {
      fprintf(stderr, "wrong image dimensions %d %d\n", rows, columns);
      return;
   }

   for (i = 0; i < rows; i++)
   {
      for (j = 0; j < columns; j++)
      {
         fread (&pixel, sizeof(pixel), 1, data_file);

         d->data.entry[i*columns + j] = ((float)(BLACK - pixel))/(float)BLACK;
         /*d->data.entry[i*rows + j] = ((float)(BLACK - pixel))/(float)BLACK;*/

      }
   }

   fread(&label, sizeof(label), 1, labels_file);
   zero_vector(&(d->label));
   d->label.entry[label] = 1.0;
   
   return;
}

/*********  MNIST DATA FUNCTIONS  ************/
#ifdef BMP
BITMAP read_bitmap_from_mnist (FILE* fp, int n)
{
   fseek (fp, 0, SEEK_SET);

   uint32_t n_images, rows, columns;

   fseek (fp, 4, SEEK_SET);
   fread (&n_images, sizeof(n_images), 1, fp);
   //n_images = (u_int32_t)be32toh(n_images);

   //assert(n >= n_images);

   fread(&rows, sizeof(rows), 1, fp);
   fread(&columns, sizeof(columns), 1, fp);

   //rows = (u_int32_t)be32toh(rows);
   //columns = (u_int32_t)be32toh(columns);

   BITMAP img = CreateEmptyBitmap(rows, columns);

   fseek (fp, 16 + n * (rows*columns), SEEK_SET);

   extract_bitmap_from_mnist (fp, img);


   return img;
}

void extract_bitmap_from_mnist (FILE* fp, BITMAP img)
{
   int i, j;
   unsigned char pixel;
   for (i = img.height -1; i >= 0; i--)
   {
      for (j = 0; j < img.width ; j++)
      {
         fread (&pixel, sizeof(pixel), 1, fp);
         PIXEL(img, i, j).blue = pixel;
         PIXEL(img, i, j).red = pixel;
         PIXEL(img, i, j).green = pixel;

      }
   }
   return;
}
#endif
vector read_vector_from_mnist (FILE* fp, int n)
{
   uint32_t n_images, rows, columns;

   fseek (fp, 4, SEEK_SET);
   fread (&n_images, sizeof(n_images), 1, fp);
   n_images = (uint32_t)be32toh(n_images);

   //assert(n >= n_images);

   fread(&rows, sizeof(rows), 1, fp);
   fread(&columns, sizeof(columns), 1, fp);

   rows = (uint32_t)be32toh(rows);
   columns = (uint32_t)be32toh(columns);

   vector img = create_vector(rows*columns);

   fseek (fp, 16 + n * (rows*columns), SEEK_SET);

   extract_vector_from_mnist (fp, &img);



   return img;
}

void extract_vector_from_mnist (FILE* fp, vector* img)
{

   int i, j;
   unsigned char pixel;
   
   
   for (i = 0; i < img->lenght; i++)
   {
      fread (&pixel, sizeof(pixel), 1, fp);
      img->entry[i] = ((float)(BLACK - pixel))/(float)BLACK;
      
   }

   return;
}

/*********  PERFORMANCE FUNCTIONS  ************/

float test_performances (neural_network n, training_data d)
{
   float result = 0;
   float r = 0;
   int i, j;
   vector pred;

   for (i = 0; i < d.n_batches; i++)
   {
      for (j = 0; j < d.batch_size; j++)
      {
         pred = prediction (n, d.data[i][j].data);

         r += standard_deviation(pred, d.data[i][j].label);

         if (max_position(pred) == max_position(d.data[i][j].label))
            result += 1;
      }
      

   }
   fprintf(stderr, "Acc: %g percent\n",result/(d.batch_size*d.n_batches)*100);
   return r/(d.n_batches*d.batch_size);

}

/*********  LOAD AND STORE FUNCTIONS  ************/


void store_neural_network (neural_network n, FILE* fp)
{
   fprintf(fp, "%d\n", n.n_layers);

   layer* l = n.input;

   fprintf(fp, "%d\n", n.input->activations.lenght);

   while ((l = l->next) != NULL)
   {
      store_matrix(fp, l->weights);
      store_vector(fp, l->biases);

   }
   return;   
}

void load_neural_network (FILE* fp, neural_network* n)
{
   int n_lay, i, input_lenght;
   layer* l = n->input;
   fscanf(fp, "%d%*c\n", &n_lay);

   if (n_lay != n->n_layers)
   {
      fprintf(stderr, "loading neural network not possible! (1)\n");
      return;
   }

   fscanf(fp, "%d%*c\n", &input_lenght);

   if (input_lenght != n->input->activations.lenght)
   {
      fprintf(stderr, "loading neural network not possible! (2)\n");
      return;
   }

   while ((l = l->next) != NULL)
   {
      load_matrix (fp, &(l->weights));
      load_vector (fp, &(l->biases));
   }

   return;
}

void store_neural_network_bin (FILE* fp, neural_network n)
{
   int i;
   /*write n layers*/
   fwrite(&n.n_layers, sizeof(n.n_layers), 1, fp);

   /*input lenght*/
   fwrite(&n.input->activations.lenght, sizeof(int), i, fp);

   layer* l = n.input;
   while ((l = l->next) != NULL)
   {
      store_layer_bin(fp, *l);
   }

   return;
}

void store_layer_bin (FILE* fp, layer l)
{
   fwrite(&l.lenght, sizeof(int), 1, fp);

   fwrite(&l.function, sizeof(int), 1, fp);

   store_matrix_bin(fp, l.weights);
   store_vector_bin(fp, l.biases);

   return;
}

neural_network load_neural_network_bin (FILE* fp)
{
   neural_network n;
   int dim_input, i;
   layer* l_prev;

   fread(&(n.n_layers), sizeof(int), 1, fp);
   fread(&(dim_input), sizeof(int), 1, fp);

   n.input = create_layer(dim_input, DEF_);
   l_prev = n.input;

   for (i = 0; i < n.n_layers; i++)
   {
      l_prev = load_layer_bin (fp);
      l_prev = l_prev->next;
   }

   n.output = l_prev;
   return n;
}

layer* load_layer_bin (FILE* fp)
{
   int dim, f;

   fread(&dim, sizeof(int), 1, fp);
   fread(&f, sizeof(int), 1, fp);

   layer* l = create_layer (dim, f);

   l->weights = load_matrix_bin(fp);
   l->biases = load_vector_bin(fp);

   return l;
}



/*ACTIVATION FUNCTIONS*/

float sigmoid (float x)
{
   float n;
   n = exp((double) -x);
   return 1.0/(1 + n);
}

float sigmoid_derivative (float x)
{
   float n = sigmoid (x);
   return exp((double) -x)*(n*n);
}

float relu (float x)
{
   if (x < 0)
      return 0.0;
   else
      return x;   
}

float relu_derivative (float x)
{
   if (x < 0)
      return 0.0;
   else
      return 1.0;  
}

float tan_h (float x)
{
   return (float)tanh(x);
}

float tanh_derivative (float x)
{
   float n = tanh(x);
   return 1 - n*n;
}

void softmax (vector* v)
{
   float div = 0.0;
   int i;

   for (i = 0; i < v->lenght; i++)
   {
      div += exp(v->entry[i]);
   }

   for (i = 0; i < v->lenght; i++)
   {
      v->entry[i] = exp(v->entry[i])/div;
   }
   return;
}

void softmax_derivative (vector *v)
{
   float x = 0;
   float p = 0;
   int i, j;

   for (i = 0; i < v->lenght; i++)
   {
      x += exp(v->entry[i]);
   }

   for (i = 0; i < v->lenght; i++)
   {
      for (j = 0; j < v->lenght; j++)
      {
         if (i == j)
         {
            p += exp(v->entry[i])/x/x *(x -exp(v->entry[i]));
         }
         else
         {
            p -= exp(v->entry[i])*exp(v->entry[j])/x/x;
         }
         

      }
      v->entry[i] = p;
   }
   return;

}
