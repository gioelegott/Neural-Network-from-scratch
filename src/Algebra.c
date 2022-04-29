#include <math.h>
#include "Algebra.h"
#include <stdio.h>
#include <stdlib.h>

/**************  BASIC MATRIX FUNCTIONS  *****************/
/* CREATE, DELETE, PRINT, READ, LOAD, STORE, RANDOMIZE, ZERO */

matrix create_matrix (int r, int c)
{
   /*creates a matrix with r rows and c columns*/

   matrix m;
   int i;

   m.rows = r;
   m.columns = c;

   /*allocating a vector of pointers*/
   m.entry = (float**)malloc(sizeof(float*)*r);

   /*allocating a vector of floats for each pointer/row*/
   for (i = 0; i < r; i++)                          
   {
      m.entry[i] = (float*)malloc(sizeof(float)*c);     
   }

   return m;
}

void delete_matrix (matrix *m)
{
   /*deallocates the memory used to store m*/
   int i;

   /*deallocating all float vectors*/
   for (i = 0; i < m->rows; i++)                
   {
      free(m->entry[i]);                                     
   }

   /*deallocating the pointer vector*/
   free(m->entry);
   
   m->rows = 0;
   m->columns = 0;

   return;
}

void print_matrix (matrix m)
{
   /*prints on terminal the matrix m*/
   int i, j;
   
   fprintf(stderr, "\n");
   for (i = 0; i < m.rows; i++)
   {
      for (j = 0; j < m.columns; j++)
         fprintf(stderr, "%g ", m.entry[i][j]);

      fprintf(stderr, "\n");
   }
   fprintf(stderr, "\n");
   return;
}

matrix read_matrix ()
{
   /*creates a matrix and reads from the command line the matrix' entry row by row*/
   int r, c, i, j;

   fprintf(stderr, "insert number of rows\n");
   scanf("%d", &r);

   fprintf(stderr, "insert number of columns\n");
   scanf("%d", &c);

   matrix m = create_matrix(r, c);

   for (i = 0; i < r; i++)
   {
      fprintf(stderr, "insert row n %d\n", i + 1);

      for (j = 0; j < c; j++)
      {
         scanf("%f", &(m.entry[i][j]));
      }
   }

   return m;
}

void store_matrix (FILE* fp, matrix m)
{
   /*stores the matrix m in text form*/
   int i, j;
   fprintf(fp, "%d %d\n", m.rows, m.columns);
   for (i = 0; i < m.rows; i++)
   {
      for (j = 0; j < m.columns; j++)
      {
         fprintf(fp, "%f", m.entry[i][j]);
      }
      fputc('\n', fp);
   }
   fputc('\n',fp);
   return;
}

void store_matrix_bin (FILE* fp, matrix m)
{
   /*stores the matrix m in binary file*/
   int i;
   fwrite(&(m.rows), sizeof(int), 1, fp);
   fwrite(&(m.columns), sizeof(int), 1, fp);
   for (i = 0; i < m.rows; i++)
   {
      fwrite(m.entry[i], sizeof(float), m.columns, fp);     
   }
   return;

}

void load_matrix (FILE* fp, matrix* m)
{
   /*loads entry from the text file fp to the matrix m*/
   int i, j, r, c;
   fscanf(fp, "%d %d%*c", &r, &c);
   if (r != m->rows || c != m->columns)
   {
      fprintf(stderr, "loading matrix not possible!\n");
      return;
   }
   
   for (i = 0; i < m->rows; i++)
   {
      for (j = 0; j < m->columns; j++)
      {
         fscanf(fp, "%f", &(m->entry[i][j]));
      }
      /*deleting \n*/
      fgetc(fp);
   }
   fgetc(fp);
   return;
}

matrix load_matrix_bin (FILE* fp)
{
   /*creates a matrix and loads entry from the binary file fp to the matrix m*/
   int i, r, c;
   fread(&r, sizeof(r), 1, fp);
   fread(&c, sizeof(c), 1, fp);
   matrix m = create_matrix(r, c);
   for (i = 0; i < m.rows; i++)
   {
      fread(m.entry[i], sizeof(float), m.columns, fp);     
   }
   return m;

}

void randomize_matrix (matrix* m, float lower, float higher)
{
   /*replaces all matrix m's entries with a random number in the interval [lower, higher]*/
   int i, j;
   srand((unsigned int)clock());

   for (i = 0; i < m->rows; i++)
   {
      for (j = 0; j < m->columns; j++)
      {
         /*randomization function*/
         m->entry[i][j] = float_random_number(lower, higher);
      }
   }
   return;
}

void zero_matrix (matrix* m)
{
   /*replaces all m's entries with 0.0*/
   int i, j;
   for (i = 0; i < m->rows; i++)
   {
      for (j = 0; j < m->columns; j++)
      {
         m->entry[i][j] = 0.0;
      }
   }
   return;
}

/**************  BASIC VECTOR FUNCTIONS  *****************/
/* CREATE, DELETE, PRINT, READ, LOAD, STORE, RANDOMIZE, ZERO */

vector create_vector (int lenght)
{
   /*creates a vector of lenght lenght*/
   vector v;
   v.entry = (float*)malloc(sizeof(float)*lenght);
   v.lenght = lenght;
   return v;
}

void delete_vector (vector* v)
{
   /*deallocates the memory of v*/
   free(v->entry);
   v->lenght = 0;
   return;
}

void print_vector (vector v)
{
   /*prints the vector v on the terminal*/
   int i;
   fprintf(stderr, "\n");
   for (i = 0; i < v.lenght; i++)
   {
      fprintf(stderr, "%g ",v.entry[i]);
   }
   fprintf(stderr, "\n");
   return;
}

vector read_vector ()
{
   /*creates and reads a vector from the terminal*/
   int l, i;
   fprintf(stderr, "insert lenght\n");
   scanf("%d", &l);
   vector v = create_vector(l);

   fprintf(stderr, "insert entry\n");
   for (i = 0; i < l; i++)
      {
         scanf("%f", &(v.entry[i]));
      }
   return v;
}

void store_vector (FILE* fp, vector v)
{
   /*stores the vector v in text file*/

   int i;
   fprintf(fp, "%d\n", v.lenght);
   for (i = 0; i < v.lenght; i++)
   {
      fprintf(fp, "%f", v.entry[i]);
   }
   fputc('\n', fp);
   return;
}

void store_vector_bin (FILE* fp, vector v)
{
   /*stores the vector v in binary file*/
   fwrite(&(v.lenght), sizeof(int), 1, fp);
   fwrite(v.entry, sizeof(float), v.lenght, fp);  

   return;

}

void load_vector (FILE* fp, vector* v)
{
   /*loads vector v from text file*/

   int i, l;

   fscanf(fp, "%d%*c", &l);

   if (l != v->lenght)
   {
      fprintf(stderr, "loading vector not possible\n");
      return;
   }

   for (i = 0; i < v->lenght; i++)
   {
      fscanf(fp, "%f", &(v->entry[i]));
   }
   fgetc(fp);
   return;
}

vector load_vector_bin (FILE* fp)
{
   /*creates and read the vector v from binary file*/
   int l;
   fread(&l, sizeof(int), 1, fp);
   vector v = create_vector(l);
   fread(v.entry, sizeof(float), v.lenght, fp);  
   return v;

}

void copy_vector (vector sour, vector* dest)
{
   /*copies the entries of sour in dest*/

   if (sour.lenght != dest->lenght)
   {
      fprintf(stderr, "copying not possible!\n");
      return;
   }

   int i;

   for (i = 0; i < sour.lenght; i++)
   {
      dest->entry[i] = sour.entry[i];
   }

}

vector create_copy_vector (vector sour)
{
   /*returns a copy of the vector sour*/

   vector dest = create_vector(sour.lenght);

   int i;
   for (i = 0; i < sour.lenght; i++)
   {
      dest.entry[i] = sour.entry[i];
   }
   return dest;
}

void randomize_vector (vector* v, float lower, float higher)
{
   /*replaces all vector v's entries with a random number in the interval [lower, higher]*/
   srand((unsigned int)clock());

   int i;
   for (i = 0; i < v->lenght; i++)
   {
      v->entry[i] = float_random_number(lower, higher);;
   }
   return;
}

void zero_vector (vector* v)
{
   /*replaces all vector v's entries with a 0.0*/

   int i;
   for (i = 0; i < v->lenght; i++)
   {
      v->entry[i] = 0.0;
   }
   return;
}

/*********  ALGEBRIC FUNCTIONS  ************/

void vector_sum (vector sour, vector* dest)
{
   /*dest += sour */
   
   if (sour.lenght != dest->lenght)
   {
      fprintf(stderr, "sum not possible\n");
      return;
   }

   int i;
   for (i = 0; i < sour.lenght; i++)
   {
      dest->entry[i] += sour.entry[i];
   }
   return;
}

void matrix_sum (matrix sour, matrix* dest)
{
   /*dest += sour */

   if(sour.rows != dest->rows || sour.columns != dest->columns)
   {
      fprintf(stderr, "sum not possible\n");
      return;
   }

   int i, j;
   for (i = 0; i < sour.rows; i++)
   {
      for (j = 0; j < sour.columns; j++)
      {
         dest->entry[i][j] += sour.entry[i][j];
      }
   }
   return;
}

void vector_subtraction (vector sour, vector* dest)
{
   /*dest -= sour */

   if(sour.lenght != dest->lenght)
   {
      fprintf(stderr, "subtraction not possible\n");
      return;
   }

   int i;
   for (i = 0; i < sour.lenght; i++)
   {
      dest->entry[i] -= sour.entry[i];
   }
   return;
}

void matrix_subtraction (matrix sour, matrix* dest)
{
   /*dest -= sour */

   if(sour.rows != dest->rows || sour.columns != dest->columns)
   {
      fprintf(stderr, "subtraction not possible\n");
      return;
   }

   int i, j;
   for (i = 0; i < sour.rows; i++)
   {
      for (j = 0; j < sour.columns; j++)
      {
         dest->entry[i][j] -= sour.entry[i][j];
      }
   }
   return;
}

void vector_function (vector *v, float (*f)(float))
{
   /*v = f(v)*/

   int i;
   for (i = 0; i < v->lenght; i++)
   {
      v->entry[i] = f(v->entry[i]);
   }
   return;
}

void matrix_function (matrix *m, float (*f)(float))
{
   /*m = f(m)*/

   int i, j;
   for (i = 0; i < m->rows; i++)
   {
      for (j = 0; j < m->columns; j++)
      {
         m->entry[i][j] = f(m->entry[i][j]);
      }
   }
   return;

}

void scalar_vector_product (vector* r, float s)
{
   /*r *= s*/

   int i;
   for (i = 0; i < r->lenght; i++)
   {     
      r->entry[i] *= s;  
   }
   return;
}

void scalar_matrix_product (matrix* r, float s)
{
   /*r *= s*/

   int i, j;
   for (i = 0; i < r->rows; i++)
   {
      for (j = 0; j < r->columns; j++)
      {
         r->entry[i][j] *= s;
      }
   }
   return;
}

float vector_dot_product (vector v1, vector v2)
{
   /*returns dot product v1 * v2*/

   if (v1.lenght != v2.lenght)
   {
      fprintf(stderr, "dot product not possible");
      return 0;
   }
   
   int i;
   float p_sum = 0;

   for (i = 0; i < v1.lenght; i++)
   {
      p_sum += v1.entry[i] * v2.entry[i];
   }

   return p_sum;
}

void matrix_product (matrix m1, matrix m2, matrix* r)
{
   /*rows times columns product r = m1 * m2*/

   if (m1.columns != m2.rows || m2.columns != r->columns || m1.rows != r->rows)
   {
      fprintf(stderr, "matrix product not possible");
      return;
   }

   int i, j, k;
   float p_sum;

   for (i = 0; i < m1.rows; i++)
   {
      for (j = 0; j < m2.columns; j++)
      {
         p_sum = 0;
         for (k = 0; k < m1.columns; k++)
         {
            p_sum += m1.entry[i][k] * m2.entry[k][j];
         }
         r->entry[i][j] = p_sum;
      }
   }
   return;
}

void matrix_vector_product (matrix m, vector v, vector* r)
{
   /*product between matrix m and verical vector v*/

   if(v.lenght != m.columns)
   {
      fprintf(stderr, "product not possible\n");
      return;
   }

   int i, j;
   float p_sum;

   for (i = 0; i < m.rows; i++)
   {
      p_sum = 0;
      for (j = 0; j < v.lenght; j++)
      {
         p_sum += v.entry[j]*m.entry[i][j];
      }
      r->entry[i] = p_sum;
   }
   return;
}

void vector_matrix_product (vector v, matrix m, vector* r)
{
   /*product between orizontal vector v and matrix m*/

   if(v.lenght != m.rows)
   {
      fprintf(stderr, "product not possible\n");
      return;
   }

   int i, j;
   float p_sum;

   for (j = 0; j < m.columns; j++)
   {
      p_sum = 0;
      for (i = 0; i < v.lenght; i++)
      {
         p_sum += v.entry[i]*m.entry[i][j];
      }
      r->entry[j] = p_sum;
   }
   return;
}

/*********  MISCELLANEOUS FUNCTIONS  ************/


int max_position (vector v)
{
   /*returns the position of the highest entry of v*/

   int i;
   int max = 0;
   for (i = 0; i < v.lenght; i++)
   {
      if (v.entry[i] > v.entry[max])
      {
         max = i;
      }
   }
   return max;
}

void vector_inverted_division (vector sour, vector* dest)
{
   /*dest = sour / dest*/

   if (sour.lenght != dest->lenght)
   {
      fprintf(stderr, "inverted division not possible\n");
      return;
   }

   int i;
   for (i = 0; i < sour.lenght; i++)
   {
      dest->entry[i] = sour.entry[i]/dest->entry[i];
   }
   return;
}

float float_random_number (float lower, float higher)
{
   /*returns a random float number in the interval [lower, higher]*/

   float r;
   float span = higher - lower;

   r = ((float)rand()/(float)RAND_MAX) * span;

   return r + lower;   
}

int int_random_number (int lower, int higher)
{
   /*returns a random int number in the interval [lower, higher]*/

   int r = (rand() + lower) % higher;
   return r;
}

float standard_deviation (vector v, vector w)
{
   vector temp = create_copy_vector(v);
   int i;
   float r = 0.0;

   vector_subtraction(w, &temp);
   vector_function (&temp, square);

   for (i = 0; i < v.lenght; i++)
   {
      r += temp.entry[i];
   }

   return sqrt(r);

}

float square (float n)
{
   return n*n;
}