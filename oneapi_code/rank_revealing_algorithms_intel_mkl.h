#include "matrix_vector_functions_intel_mkl.h"

void low_rank_svd_decomp_fixed_rank_or_prec(mat *M, myint64 k, double TOL, myint64 *frank, mat **U, mat **S, mat **V);
void low_rank_svd_rand_decomp_fixed_rank(mat *M, myint64 k, myint64 p, myint64 vnum, myint64 q, myint64 s, myint64 *frank, mat **U, mat **S, mat **V);

void pivotedQR_mkl(mat *M, mat **Q, mat **R, vec **I);
void get_householder_matrix(vec *x, myint64 ind1, myint64 ind2, mat *H);

void use_low_rank_svd_for_approximation(mat *M, mat *U, mat *S, mat *V);
void use_pivoted_QR_decomp_for_approximation(mat *M, mat *Qk, mat *Rk, vec *I);
