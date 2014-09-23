/*
 * cula gpu code with host openmp 
 */
#include "low_rank_svd_algorithms_nvidia_cula.h"


int main(int argc, char** argv){
    int i, j, m, n, k, culaVersion;
    double normM,normU,normS,normV,normP,percent_error;
    mat *M, *U, *S, *V, *P;
    time_t start_time, end_time;
    char *M_file = "../data/A_mat1.bin";

    culaStatus status;

    printf("Initializing CULA\n");
    status = culaInitialize();
    checkStatus(status);

    culaVersion = culaGetVersion();
    printf("culaVersion is %d\n", culaVersion);
    
    printf("loading matrix from %s\n", M_file);
    M = matrix_load_from_binary_file(M_file);
    m = M->nrows;
    n = M->ncols;
    printf("sizes of M are %d by %d\n", m, n);


    // now test low rank SVD of M..
    k = 1000;
    U = matrix_new(m,k);
    S = matrix_new(k,k);
    V = matrix_new(n,k);
    
    printf("calling random SVD with k = %d\n", k);
    time(&start_time);
    //randomized_low_rank_svd1(M, k, U, S, V);
    //randomized_low_rank_svd2(M, k, U, S, V);
    randomized_low_rank_svd3(M, k, 20, U, S, V);
    time(&end_time);
    printf("elapsed time: about %d seconds\n", (int)difftime(end_time,start_time));

    // form product matrix
    P = matrix_new(m,n);
    form_svd_product_matrix(U,S,V,P);

    // get norms of each
    normM = matrix_frobenius_norm(M);
    normU = matrix_frobenius_norm(U);
    normS = matrix_frobenius_norm(S);
    normV = matrix_frobenius_norm(V);
    normP = matrix_frobenius_norm(P);
    printf("normM = %f ; normU = %f ; normS = %f ; normV = %f ; normP = %f\n", normM, normU, normS, normV, normP);

    // calculate percent error
    percent_error = get_percent_error_between_two_mats(M,P);
    printf("percent_error between M and U S V^T = %f\n", percent_error);


    // delete and exit
    matrix_delete(M);
    matrix_delete(U);
    matrix_delete(S);
    matrix_delete(V);
    matrix_delete(P);
 

    printf("Shutting down CULA\n");
    culaShutdown();

    return EXIT_SUCCESS;
}

