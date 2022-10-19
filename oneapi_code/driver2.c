/*  
   driver 2: test block random QB routine and subsequent low rank SVD 
*/

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

#include "rank_revealing_algorithms_one_api.h"
#include <stdint.h>
#include <inttypes.h>
#include <omp.h>


int main()
{
    myint64 i, j, m, n, kstep, nstep, p, q, s, offset;
    myint64 frank, numnnz;
    float val,normM,normU,normS,normV,normP,percent_error;
    mat *M, *D, *MD, *MDMT, *Q, *B, *U, *S, *V;
    vec *svals;
    double start_time, end_time;

    m = 8*1500;
    n = 8*2000;
    numnnz = m*n; // dense

    M = matrix_new(m,n);
    m = M->nrows;
    n = M->ncols;
    printf("sizes of M are %ld by %ld\n", m, n);
    printf("m*n = %" PRId64 "\n", numnnz); // use 64 bit int 

    printf("initializing random matrix..\n");
    printf("--- calling initialize random matrix ---\n");
    start_time = dsecnd(); // or use omp_get_wtime
    initialize_random_matrix(M);
    end_time = dsecnd();
    printf("--- done with initialization ---\n");
    printf("elapsed time: about %4.2f seconds\n", get_seconds_frac(start_time,end_time));


    printf("compute MDM^T (mirror QDQ^T) --\n");
    //can call QR_factorization_getQ(M,Q);
    start_time = omp_get_wtime();
    D = matrix_new(n,n);
    svals = vector_new(n);
    printf("set svals to decay\n");
    vector_set_element(svals,0,1.0);
    for(i=1; i<n; i++){
        vector_set_element(svals,i, vector_get_element(svals,i-1)/1.02);
    }
    initialize_diagonal_matrix(D, svals);
    MD = matrix_new(m,n);
    MDMT = matrix_new(m,m);
    printf("compute MD\n");
    matrix_matrix_mult(M,D,MD);
    printf("compute MDMT\n");
    matrix_matrix_transpose_mult(MD,M,MDMT);
    matrix_delete(M);
    matrix_delete(MD);
    end_time = omp_get_wtime();;
    printf("elapsed time: about %4.2f seconds\n", end_time - start_time);

    // params for QB, determining SVD..
    printf("computing approximate low rank SVD via QB\n");
    kstep = 100; // block size 
    nstep = 10; // num steps
    p = 2; // power scheme
    s = 1; // re-rotho for power scheme    

    printf("calling random QB..\n");
    start_time = omp_get_wtime();
    randQB_pb(MDMT, kstep, nstep, p, s, &Q, &B);
    end_time = omp_get_wtime();
    printf("elapsed time: about %4.2f seconds\n", end_time - start_time);

    printf("evaluating approximation\n");
    start_time = omp_get_wtime();
    use_QB_decomp_for_approximation(MDMT, Q, B);
    end_time = omp_get_wtime();
    printf("elapsed time: about %4.2f seconds\n", end_time - start_time);

    printf("using QB for SVD..\n");
    start_time = omp_get_wtime();
    low_rank_svd_rand_decomp_fromQB(&Q, &B, &U, &S, &V);
    end_time = omp_get_wtime();
    printf("elapsed time: about %4.2f seconds\n", end_time - start_time);

    printf("evaluating approximation\n");
    matrix_delete(Q);
    matrix_delete(B);
    normM = get_matrix_frobenius_norm(MDMT);
    normU = get_matrix_frobenius_norm(U);
    normS = get_matrix_frobenius_norm(S);
    normV = get_matrix_frobenius_norm(V);
    printf("normM = %f ; normU = %f ; normS = %f ; normV = %f\n", normM, normU, normS, normV);

    use_low_rank_svd_for_approximation(MDMT, U, S, V);

    // delete and exit
    printf("delete and exit..\n");
    matrix_delete(MDMT);
    matrix_delete(U);
    matrix_delete(S);
    matrix_delete(V);

    return 0;
}
