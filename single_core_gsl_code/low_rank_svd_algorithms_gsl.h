#include "matrix_vector_functions_gsl.h"


/* computes the approximate low rank SVD of rank k of matrix M using BBt version */
void randomized_low_rank_svd1(gsl_matrix *M, int k, gsl_matrix *U, gsl_matrix *S, gsl_matrix *V);


/* computes the approximate low rank SVD of rank k of matrix M using QR method */
void randomized_low_rank_svd2(gsl_matrix *M, int k, gsl_matrix *U, gsl_matrix *S, gsl_matrix *V);


/* computes the approximate low rank SVD of rank k of matrix M using QR method and 
(M M^T)^q M R sampling */
void randomized_low_rank_svd3(gsl_matrix *M, int k, int q, int s, gsl_matrix *U, gsl_matrix *S, gsl_matrix *V);

