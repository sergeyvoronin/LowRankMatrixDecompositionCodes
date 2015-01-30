/* single core code using GSL library */
#include "low_rank_svd_algorithms_gsl.h"


int main (void)
{
    int i, j, m, n, k;
    double percent_error, normM, normU, normS, normV, normP;
    time_t start_time, end_time;
    //char *mfile = "../data/A_mat1.bin";
    char *mfile = "../data/A_mat_2kx4k.bin";
    gsl_matrix *U,*S,*V;
    
    // low rank svd rank
    k = 300;
    //k = 500;

    // load matrix
    printf("loading matrix from %s\n", mfile);
    gsl_matrix *M = matrix_load_from_binary_file(mfile);
    m = M->size1; n = M->size2;
    printf("sizes of M are %d by %d\n", m,n);

        // call random SVD
    printf("calling random SVD..\n");
    time(&start_time);
    //randomized_low_rank_svd1(M, k, &U, &S, &V);
    //randomized_low_rank_svd2(M, k, &U, &S, &V);
    //randomized_low_rank_svd3(M, k, 3, 1, &U, &S, &V);
    randomized_low_rank_svd2_autorank1(M, 0.5, 0.01, &U, &S, &V);
    time(&end_time);
    printf("elapsed time: about %d seconds\n", (int)difftime(end_time,start_time));

    // form product matrix
    gsl_matrix *P = gsl_matrix_alloc(m,n);
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

    // write to disk
    /*printf("writing results to disk..\n");
    matrix_write_to_binary_file(U, "../data/output/single_core/U.bin");
    matrix_write_to_binary_file(S, "../data/output/single_core/S.bin");
    matrix_write_to_binary_file(V, "../data/output/single_core/V.bin");
    printf("finished\n");*/

    // free matrices
    gsl_matrix_free(M);
    gsl_matrix_free(U);
    gsl_matrix_free(S);
    gsl_matrix_free(V);
    gsl_matrix_free(P);

    return 0;
}

