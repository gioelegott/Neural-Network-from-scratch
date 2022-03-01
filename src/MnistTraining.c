#include "NeuralNetwork.h"
/*in extract labeled data bisogna cambiare!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
int main()
{

   int v[3] = {DATA_SIZE, 50, 50}; //<---- vector that describes the neural network structure
   float momentum = 0.0;
   float learning_rate = 0.01;
   int epochs = 10;


   float accuracy;
   
   printf("Creating neural network...");
   neural_network n = create_neural_network(3, v, SIGMOID);
   add_layer(&n, LABEL_SIZE, SIGMOID);

   randomize_network(&n);
   printf("   done!\n");

   /*training data*/
   printf("Loading training data...");
   FILE* fp_d_train = fopen("../training/train-images-idx3-ubyte", "rb");
   if (fp_d_train == NULL)
      printf("\nfile 1 non aperto\n");   


   FILE* fp_l_train = fopen("../training/train-labels-idx1-ubyte", "rb");
   if (fp_l_train == NULL)
      printf("\nfile 2 non aperto\n");
   
   training_data data = load_training_data(fp_d_train, fp_l_train, 60);  
   printf("   done!\n");

   /*testing data*/
   printf("Loading testing data...");
   FILE* fp_d_test = fopen("../training/t10k-images-idx3-ubyte", "rb");
   if (fp_d_test == NULL)
      printf("file 3 non aperto\n");

   FILE* fp_l_test = fopen("../training/t10k-labels-idx1-ubyte", "rb");
   if (fp_l_test == NULL)
      printf("file 4 non aperto\n");
   
   training_data test_data = load_training_data(fp_d_test, fp_l_test, 1000);  
   printf("   done!\n");

   printf("TRAINING STARTED...\n");


   train(&n, data, epochs, learning_rate, momentum, test_data, 1);
   printf("TRAINING FINISHED!\n\n");

   FILE* fp_storage = fopen("../saves/h2_50_20-e10-acc45.nnb", "wb");

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