#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct
{
   int rows;
   int columns;
   float **entry;
} matrix;

typedef struct
{
   int lenght;
   float *entry;
} vector;

/*matrix management functions*/
matrix create_matrix (int r, int c);
void delete_matrix (matrix *m);
void print_matrix (matrix m);
matrix read_matrix ();
void load_matrix (FILE* fp, matrix* m);
matrix load_matrix_bin (FILE* fp);
void store_matrix (FILE* fp, matrix m);
void store_matrix_bin (FILE* fp, matrix m);
void randomize_matrix (matrix* m, float lower, float higher);
void zero_matrix (matrix* m);

/*vector management functions*/
vector create_vector (int lenght);
void delete_vector (vector* v);
void print_vector (vector v);
vector read_vector ();
void load_vector (FILE* fp, vector* v);
vector load_vector_bin (FILE* fp);
void store_vector (FILE* fp, vector v);
void store_vector_bin (FILE* fp, vector v);
vector create_copy_vector (vector sour);
void copy_vector (vector sour, vector* dest);
void randomize_vector (vector* v, float lower, float higher);
void zero_vector (vector* v);

/*algebric functions*/
void vector_sum (vector sour, vector* dest);
void matrix_sum (matrix sour, matrix* dest);
void vector_subtraction (vector sour, vector* dest);
void matrix_subtraction (matrix sour, matrix* dest);
void vector_function (vector *v, float (*f)(float));
void matrix_function (matrix *m, float (*f)(float));
void scalar_vector_product (vector* r, float s);
void scalar_matrix_product (matrix* r, float s);
float vector_dot_product (vector v1, vector v2);
void matrix_product (matrix m1, matrix m2, matrix* r);
void matrix_vector_product (matrix m, vector v, vector* r);
void vector_matrix_product (vector v, matrix m, vector* r);

/*miscellaneous functions*/
int max_position (vector v);
void vector_inverted_division (vector sour, vector* dest);
float float_random_number (float lower, float higher);
int int_random_number (int lower, int higher);
float standard_deviation (vector v, vector w);
float square (float n);



