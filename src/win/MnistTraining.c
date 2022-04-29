#include "NeuralNetwork.h"
/*in extract labeled data bisogna cambiare!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
int main()
{
   int v[3] = {DATA_SIZE, 20};
   float accuracy;
   //vector a = create_vector(30);
   //randomize_vector (&a, -1, 1);

   fprintf(stderr, "Creating neural network...");
   neural_network n = create_neural_network(2, v, SIGMOID);
   add_layer(&n, LABEL_SIZE, SIGMOID);

   randomize_network(&n);
   fprintf(stderr, "   done!\n");

   /*training data*/
   fprintf(stderr, "Loading training data...");
   FILE* fp_d_train = fopen("C:/Users/Gioele/Desktop/c/train-images-idx3-ubyte", "rb");
   if (fp_d_train == NULL)
      fprintf(stderr, "\nfile 1 non aperto\n");   


   FILE* fp_l_train = fopen("C:/Users/Gioele/Desktop/c/train-labels-idx1-ubyte", "rb");
   if (fp_l_train == NULL)
      fprintf(stderr, "\nfile 2 non aperto\n");
   
   training_data data = load_training_data(fp_d_train, fp_l_train, 60);  
   fprintf(stderr, "   done!\n");

   /*testing data*/
   fprintf(stderr, "Loading testing data...");
   FILE* fp_d_test = fopen("C:/Users/Gioele/Desktop/c/t10k-images-idx3-ubyte", "rb");
   if (fp_d_test == NULL)
      fprintf(stderr, "file 3 non aperto\n");

   FILE* fp_l_test = fopen("C:/Users/Gioele/Desktop/c/t10k-labels-idx1-ubyte", "rb");
   if (fp_l_test == NULL)
      fprintf(stderr, "file 4 non aperto\n");
   
   training_data test_data = load_training_data(fp_d_test, fp_l_test, 1000);  
   fprintf(stderr, "   done!\n");

   fprintf(stderr, "TRAINING STARTED...\n");
   /*10 epochs*/
   float momentum = 0.0;
   float learning_rate = 0.01;
   int epochs = 8;
   train(&n, data, epochs, learning_rate, momentum, test_data, 1);
   fprintf(stderr, "TRAINING FINISHED!\n\n");

   FILE* fp_storage = fopen("C:/Users/Gioele/Desktop/c/mnist_network.nnb", "wb");

   store_neural_network_bin (fp_storage, n);

   fclose (fp_storage);
   fclose(fp_d_train);
   fclose(fp_l_train);
   fclose(fp_d_test);
   fclose(fp_l_test);


   delete_neural_network(&n);

   delete_training_data(&data);

   delete_training_data(&test_data);

   return EXIT_SUCCESS;
}