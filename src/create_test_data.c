#include "algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void store_array (FILE* fp, float a[], int dim);
void create_data (float data[], int dim);
void create_label (float label[], int l_size, float data[], int d_size);


int main ()
{
   FILE *fp_d, *fp_l;
   int n_files = 10000;
   int data_size = 15;
   int label_size = 3;

   char name[32];

   float data[data_size];
   float label[label_size];



   for (int i = 0; i < n_files; i++)
   {
      sprintf(name, "test%d.dat", i);
      fp_d = fopen(name, "w");
      create_data(data, data_size);
      store_array(fp_d, data, data_size);

      sprintf(name, "test%d.lbl", i);
      fp_l = fopen(name, "w");
      create_label(label, label_size, data, data_size);
      store_array(fp_l, label, label_size);

      fclose(fp_d);
      fclose(fp_l);

   }
   return EXIT_SUCCESS;
}

void store_array (FILE* fp, float a[], int dim)
{
   int i;
   for (i = 0; i < dim; i++)
   {
      fprintf(fp, "%f ", a[i]);
   }
   return;
}

void create_data (float data[], int dim)
{
   int i;
   srand(clock());
   for (i = 0; i < dim; i++)
   {
      data[i] = rand()/(float)RAND_MAX;
   }
   return;
}

void create_label (float label[], int l_size, float data[], int d_size)
{
   int i;
   int f;
   
   for (i = 0; i < l_size; i++)
   {
      f += (int)(data[i+2] * data[i+3] * 5);
      label[i] = 0;
   }

   label[f%l_size] = 1.0;

   return;
}
