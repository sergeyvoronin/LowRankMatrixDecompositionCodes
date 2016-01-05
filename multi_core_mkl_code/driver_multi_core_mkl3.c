/* Intel MKL code with OpenMP : test driver for interpolative decomposition routines */

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

#include "rank_revealing_algorithms_intel_mkl.h"

int main()
{
    int i, j, m, n, k;
    double normM,normU,normS,normV,normP,percent_error;
    mat *M, *T, *S;
    vec *Icol, *Irow;
    time_t start_time, end_time;
    //char *M_file = "../data/A_mat_6kx12k.bin";
    char *M_file = "../data/A_mat_2kx4k.bin";

    printf("loading matrix from %s\n", M_file);
    M = matrix_load_from_binary_file(M_file);
    m = M->nrows;
    n = M->ncols;
    printf("sizes of M are %d by %d\n", m, n);

    // now test rank k ID of M..
    k = 400;
    
    printf("calling rank %d column ID routine..\n", k);
    time(&start_time);
    id_decomp_fixed_rank(M, k, &Icol, &T);
    time(&end_time);
    printf("elapsed time: about %d seconds\n", (int)difftime(end_time,start_time));
    
    printf("check error\n");
    use_id_decomp_for_approximation(M, T, Icol, k);

    printf("calling rank %d two sided ID routine..\n", k);
    time(&start_time);
    id_two_sided_decomp_fixed_rank(M, k, &Icol, &Irow, &T, &S);
    time(&end_time);
    printf("elapsed time: about %d seconds\n", (int)difftime(end_time,start_time));
    
    printf("check error\n");
    use_id_two_sided_decomp_for_approximation(M, T, S, Icol, Irow, k);

    // delete and exit
    printf("delete and exit..\n");
    matrix_delete(M); matrix_delete(T); matrix_delete(S);
    vector_delete(Icol); vector_delete(Irow);

    return 0;
}

