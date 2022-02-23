#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "algebra.h"

int main ()
{
    uint32_t n = 6421740;
    
    char c1 = n>>24;
    char c2 = n<<24;
    char c3 = n>>8;
    char c4 = n<<8;

    uint32_t m = c1&0xff | c2&0xff000000 | c3&0xff00 | c4&0xff0000;

    printf("%d\n", m);

    vector v = create_vector(5);
    randomize_vector(&v, 0.0, 1.0);
    print_vector(v);
    delete_vector(&v);

    return EXIT_SUCCESS;
}