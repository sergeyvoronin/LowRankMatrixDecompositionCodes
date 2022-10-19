/*  
   driver 1: test low rank SVD routine for a large matrix
*/

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

#include "rank_revealing_algorithms_one_api.h"
#include <stdint.h>
#include <inttypes.h>

int main()
{
    myint64 i, j, m, n, k, p, q, s, vnum, offset;
    myint64 frank, numnnz;
    float val,normM,normU,normS,normV,normP,percent_error;
    mat *M, *D, *MD, *MDMT, *U, *S, *V, *P;
    vec *svals;
    double start_time, end_time;

    m = 25000;
    n = 30000;
    numnnz = m*n; // dense

    M = matrix_new(m,n);
    m = M->nrows;
    n = M->ncols;
    printf("sizes of M are %ld by %ld\n", m, n);
    printf("m*n = %" PRId64 "\n", numnnz); // this is bigger than a 32 bit int can hold

    printf("initializing random matrix..\n");
    printf("--- calling initialize random matrix ---\n");
    start_time = dsecnd();
    initialize_random_matrix(M);
    end_time = dsecnd();
    printf("--- done with initialization ---\n");
    printf("elapsed time: about %4.2f seconds\n", get_seconds_frac(start_time,end_time));


    printf("compute MDM^T (mirror QDQ^T) --\n");
    //can call QR_factorization_getQ(M,Q);
    start_time = dsecnd();
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
    end_time = dsecnd();
    printf("elapsed time: about %4.2f seconds\n", get_seconds_frac(start_time,end_time));

    // now test low rank SVD of M..
    printf("computing approximate low rank SVD\n");
    k = 1000; // rank we want
    p = 20; // oversampling
    q = 3; // power scheme
    s = 1; // re-rotho for power scheme    
    vnum = 1; // scheme to use

    printf("calling random SVD..\n");
    start_time = dsecnd();
	low_rank_svd_rand_decomp_fixed_rank(MDMT, k, p, vnum, q, s, &frank, &U, &S, &V);
    end_time = dsecnd();
    printf("elapsed time: about %4.2f seconds\n", get_seconds_frac(start_time,end_time));

    // form product matrix
    P = matrix_new(m,n);
    form_svd_product_matrix(U,S,V,P);

    // get norms of each
    normM = get_matrix_frobenius_norm(MDMT);
    normU = get_matrix_frobenius_norm(U);
    normS = get_matrix_frobenius_norm(S);
    normV = get_matrix_frobenius_norm(V);
    normP = get_matrix_frobenius_norm(P);
    printf("normM = %f ; normU = %f ; normS = %f ; normV = %f ; normP = %f\n", normM, normU, normS, normV, normP);

    // calculate percent error
    percent_error = get_percent_error_between_two_mats(MDMT,P);
    printf("percent_error between MDMT and U S V^T = %f\n", percent_error);

    // delete and exit
    printf("delete and exit..\n");
    matrix_delete(MDMT);
    matrix_delete(U);
    matrix_delete(S);
    matrix_delete(V);
    matrix_delete(P);

    return 0;
}

