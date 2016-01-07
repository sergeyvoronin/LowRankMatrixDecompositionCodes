#include "matrix_vector_functions_intel_mkl.h"


/* computes the approximate low rank SVD of rank k of matrix M using BBt version */
void randomized_low_rank_svd1(mat *M, int k, mat **U, mat **S, mat **V);


/* computes the approximate low rank SVD of rank k of matrix M using QR version */
void randomized_low_rank_svd2(mat *M, int k, mat **U, mat **S, mat **V);


/* computes the approximate low rank SVD of rank k of matrix M using QR version 
 * with range sampling via (M M^T)^q M R*/
void randomized_low_rank_svd3(mat *M, int k, int q, int s, mat **U, mat **S, mat **V);

/* computes the approximate low rank SVD of rank k of matrix M using the 
QB blocked algorithm for Q and BBt method */
void randomized_low_rank_svd4(mat *M, int kstep, int nstep, int q, mat **U, mat **S, mat **V);


/* version 2 of ramdomized low rank SVD with autorank estimation 1 */
void randomized_low_rank_svd2_autorank1(mat *M, double frac_of_max_rank, double TOL, mat **U, mat **S, mat **V);


/* version 2 of ramdomized low rank SVD with autorank estimation 2 */
void randomized_low_rank_svd2_autorank2(mat *M, int kblocksize, double TOL, mat **U, mat **S, mat **V);


/* version 3 of ramdomized low rank SVD with autorank estimation 2 */
void randomized_low_rank_svd3_autorank2(mat *M, int kblocksize, double TOL, int q, int s, mat **U, mat **S, mat **V);


/* full pivoted QR with intel mkl library */
void pivotedQR_mkl(mat *M, mat **Q, mat **R, vec **I);

/* householder matrix implementation */
void get_householder_matrix(vec *x, int ind1, int ind2, mat *H);

/* partial pivoted qr via householder  - faster version, no P */
void pivoted_QR_of_specified_rank(mat *M, int k, int *frank, mat **Qk, mat **Rk, vec **I);



/* randQB algorithm one vector at a time */
void randQB_p(mat *M, int k, int p, mat **Q, mat **B);

/* randQB algorithm blocked */
void randQB_pb(mat *M, int kstep, int nstep, int p, mat **Q, mat **B);

/* solve A X = B where A is upper triangular matrix and X is a matrix 
invert different ways
1. using tridiagonal matrix system solve
2. using inverse of tridiagonal matrix solve
3. Use SVD of A to compute inverse 
default: solve column by column with tridiagonal system
*/
//void upper_triangular_system_solve(mat *A, mat *B, mat *X, int solve_type);


/* computes the column ID decomposition of a matrix of specified rank 
: [I,T] = id_decomp_fixed_rank(M,k) 
where I is the vector from the permutation and T = inv(Rk1)*Rk2 */
void id_decomp_fixed_rank(mat *M, int k, vec **I, mat **T);

/* computes two sided ID decomposition of a matrix of specified rank */
void id_two_sided_decomp_fixed_rank(mat *M, int k, vec **Icol, vec **Irow, mat **T, mat **S);

/* computes a rank k cur decomposition of a matrix */
void cur_decomp_fixed_rank(mat *M, int k, mat **C, mat **U, mat **R);


/* evaluate approximation to M using supplied low rank SVD of rank k */
void use_low_rank_svd_for_approximation(mat *M, mat *U, mat *S, mat *V);

/* evaluate approximation to M using supplied column ID of rank k */
void use_id_decomp_for_approximation(mat *M, mat *T, vec *I, int k);

/* evaluate approximation to M using supplied two sided ID of rank k */
void use_id_two_sided_decomp_for_approximation(mat *M, mat *T, mat *S, vec *Icol, vec *Irow, int k);
 
/* evaluate approximation to M using supplied CUR decomposition of rank k */
void use_cur_decomp_for_approximation(mat *M, mat *C, mat *U, mat *R);

