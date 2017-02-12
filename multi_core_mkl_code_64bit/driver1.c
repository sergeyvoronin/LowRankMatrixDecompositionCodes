/* Intel MKL code with OpenMP 
   driver 1: test low rank SVD routine for a large matrix
*/

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

#include "rank_revealing_algorithms_intel_mkl.h"
#include <stdint.h>
#include <inttypes.h>

int main()
{
    int i, j, m, n, k, p, q, s, vnum, offset;
    int64_t frank, numnnz;
    double val,normM,normU,normS,normV,normP,percent_error;
    mat *M, *U, *S, *V, *P;
    struct timeval start_time, end_time;

    m = 50000;
    n = 100000;
    numnnz = m*n; // dense

    M = matrix_new(m,n);
    m = M->nrows;
    n = M->ncols;
    printf("sizes of M are %d by %d\n", m, n);
    printf("m*n = %" PRId64 "\n", numnnz); // this is bigger than a 32 bit int can hold

    printf("initializing random matrix..\n");
    gettimeofday(&start_time,NULL);

    initialize_random_matrix(M);

    gettimeofday(&end_time,NULL);
    printf("done loading..\n");
    printf("elapsed time: about %4.2f seconds\n", get_seconds_frac(start_time,end_time));


    // now test low rank SVD of M..
    k = 1000; // rank we want
    p = 20; // oversampling
    q = 3; // power scheme
    s = 1; // re-rotho for power scheme    
    vnum = 1; // scheme to use

    printf("calling random SVD..\n");
    gettimeofday(&start_time,NULL);
	low_rank_svd_rand_decomp_fixed_rank(M, k, p, vnum, q, s, &frank, &U, &S, &V);
    gettimeofday(&end_time,NULL);
    printf("elapsed time: about %4.2f seconds\n", get_seconds_frac(start_time,end_time));

    // form product matrix
    P = matrix_new(m,n);
    form_svd_product_matrix(U,S,V,P);

    // get norms of each
    normM = get_matrix_frobenius_norm(M);
    normU = get_matrix_frobenius_norm(U);
    normS = get_matrix_frobenius_norm(S);
    normV = get_matrix_frobenius_norm(V);
    normP = get_matrix_frobenius_norm(P);
    printf("normM = %f ; normU = %f ; normS = %f ; normV = %f ; normP = %f\n", normM, normU, normS, normV, normP);

    // calculate percent error
    percent_error = get_percent_error_between_two_mats(M,P);
    printf("percent_error between M and U S V^T = %f\n", percent_error);

    // delete and exit
    printf("delete and exit..\n");
    matrix_delete(M);
    matrix_delete(U);
    matrix_delete(S);
    matrix_delete(V);
    matrix_delete(P);

    return 0;
}

