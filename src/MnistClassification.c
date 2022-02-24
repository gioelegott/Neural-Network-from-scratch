#include "NeuralNetwork.h"


int main (int argc, char* argv[])
{
   char label;
   int i, r;
   if (argc != 2)
   {
      printf("syntax: -./mnist_classification n-\nwhere n is the index of the chosen image\n");
      return EXIT_FAILURE;
   }

   /*loading neural network*/
   FILE* fp_n = fopen("C:/Users/Gioele/Desktop/c/mnist_network200__2.nn", "r");

   int v[3] = {DATA_SIZE, 200, LABEL_SIZE};
   neural_network net = load_neural_network_bin(fp_n);

   printf("loaded neural network\n");

   /*opening mnist data as bmp*/
   int n = atoi(argv[1]);

   FILE* fp = fopen("C:/Users/Gioele/Desktop/c/t10k-images-idx3-ubyte", "rb");
   FILE* fp_lbl = fopen("C:/Users/Gioele/Desktop/c/t10k-labels-idx1-ubyte", "rb");
   FILE* f_img = fopen("C:/Users/Gioele/Desktop/c/number.bmp", "wb");

   //BITMAP img = read_bitmap_from_mnist (fp, n);

   //WriteBitmap (img, f_img);
   //ReleaseBitmapData (&img);

   /*neural network prediction*/
   fseek(fp_lbl, 8 + n, SEEK_SET);
   fread(&label, sizeof(label), 1, fp_lbl);

   vector data = read_vector_from_mnist (fp, n);
   vector pr = prediction (net, data);
   //softmax(&pr);
   //print_vector(pr);
   printf("Prediction: %d\nLabel: %d\n\n", max_position(pr), (int)label);
   delete_vector (&data);
   
   rewind(fp_lbl);
   rewind(fp);
   
   training_data test_data = load_training_data(fp, fp_lbl, 1000);

   test_performances(net, test_data);

   delete_training_data(&test_data);

   delete_neural_network (&net);
   fclose(fp);
   fclose(f_img);
   fclose(fp_n);

   return EXIT_SUCCESS;
}