#include "matrix_vector_functions_intel_mkl.h"

void low_rank_svd_decomp_fixed_rank_or_prec(mat *M, int64_t k, double TOL, int64_t *frank, mat **U, mat **S, mat **V);

void low_rank_svd_rand_decomp_fixed_rank(mat *M, int64_t k, int64_t p, int64_t vnum, int64_t q, int64_t s, int64_t *frank, mat **U, mat **S, mat **V);

void low_rank_svd_blockrand_decomp_fixed_rank_or_prec(mat *M, int64_t k, int64_t p, double TOL, 
    int64_t vnum, int64_t kstep, int64_t q, int64_t s, int64_t *frank, mat **U, mat **S, mat **V);


/* computes the approximate low rank SVD of rank k of matrix M using BBt version */
void randomized_low_rank_svd1(mat *M, int64_t k, mat **U, mat **S, mat **V);


/* computes the approximate low rank SVD of rank k of matrix M using QR version */
void randomized_low_rank_svd2(mat *M, int64_t k, mat **U, mat **S, mat **V);


/* computes the approximate low rank SVD of rank k of matrix M using QR version 
 * with range sampling via (M M^T)^q M R*/
void randomized_low_rank_svd3(mat *M, int64_t k, int64_t q, int64_t s, mat **U, mat **S, mat **V);

/* computes the approximate low rank SVD of rank k of matrix M using the 
QB blocked algorithm for Q and BBt method */
void randomized_low_rank_svd4(mat *M, int64_t kstep, int64_t nstep, int64_t s, mat **U, mat **S, mat **V);


/* version 2 of ramdomized low rank SVD with autorank estimation 1 */
void randomized_low_rank_svd2_autorank1(mat *M, double frac_of_max_rank, double TOL, mat **U, mat **S, mat **V);


/* version 2 of ramdomized low rank SVD with autorank estimation 2 */
void randomized_low_rank_svd2_autorank2(mat *M, int64_t kblocksize, double TOL, mat **U, mat **S, mat **V);


/* version 3 of ramdomized low rank SVD with autorank estimation 2 */
void randomized_low_rank_svd3_autorank2(mat *M, int64_t kblocksize, double TOL, int64_t q, int64_t s, mat **U, mat **S, mat **V);


/* full pivoted QR with intel mkl library */
void pivotedQR_mkl(mat *M, mat **Q, mat **R, vec **I);

/* householder matrix implementation */
void get_householder_matrix(vec *x, int64_t ind1, int64_t ind2, mat *H);

/* partial pivoted qr via householder  - faster version, no P */
void pivoted_QR_of_specified_rank(mat *M, int64_t k, int64_t *frank, mat **Qk, mat **Rk, vec **I);

/* partial pivoted qr via householder of specified rank or precision */
void pivoted_QR_of_specified_rank_or_prec(mat *M, int64_t k, double TOL, int64_t *frank, mat **Qk, mat **Rk, vec **I);


/* randQB algorithm one vector at a time */
void randQB_p(mat *M, int64_t k, int64_t p, mat **Q, mat **B);

/* randQB algorithm blocked */
void randQB_pb(mat *M, int64_t kstep, int64_t nstep, int64_t q, int64_t s, mat **Q, mat **B);

/* randQB algorithm blocked */
void randQB_pb_new(mat *M, int64_t kstep, int64_t nstep, double TOL, int64_t q, int64_t s, int64_t *frank, mat **Q, mat **B);


/* computes the column ID decomposition of a matrix of specified rank 
where I is the vector from the permutation and T = inv(Rk1)*Rk2 */
void id_decomp_fixed_rank_or_prec(mat *M, int64_t k, double TOL, int64_t *frank, vec **I, mat **T);

/* randomized ID of rank k */
void id_rand_decomp_fixed_rank(mat *M, int64_t k, int64_t p, int64_t q, int64_t s, vec **I, mat **T);

/* block randomized ID of rank k */
void id_blockrand_decomp_fixed_rank_or_prec(mat *M, int64_t k, int64_t p, double TOL, int64_t kstep, int64_t q, int64_t s, int64_t *frank, vec **I, mat **T);


/* computes two sided ID decomposition of a matrix of specified rank */
void id_two_sided_decomp_fixed_rank_or_prec(mat *M, int64_t k, double TOL, int64_t *frank, vec **Icol, vec **Irow, mat **T, mat **S);

/* randomized two sided ID of rank k */
void id_two_sided_rand_decomp_fixed_rank(mat *M, int64_t k, int64_t l, int64_t p, int64_t s, vec **Icol, vec **Irow, mat **T, mat **S);

/* block randomized two sided ID of rank k */
void id_two_sided_blockrand_decomp_fixed_rank_or_prec(mat *M, int64_t k, int64_t p, double TOL, int64_t kstep, int64_t q, int64_t s, int64_t *frank, vec **Icol, vec **Irow, mat **T, mat **S);


/* computes a rank k cur decomposition of a matrix */
void cur_decomp_fixed_rank_or_prec(mat *M, int64_t k, double TOL, int64_t *frank, mat **C, mat **U, mat **R);

/* computes a randomized rank k cur decomposition of a matrix */
void cur_rand_decomp_fixed_rank(mat *M, int64_t k, int64_t p, int64_t q, int64_t s, mat **C, mat **U, mat **R);

/* computes a block randomized rank k cur decomposition of a matrix */
void cur_blockrand_decomp_fixed_rank_or_prec(mat *M, int64_t k, int64_t p, double TOL, int64_t kstep, int64_t q, int64_t s, int64_t *frank, mat **C, mat **U, mat **R);


/* evaluate approximation to M using supplied low rank SVD of rank k */
void use_low_rank_svd_for_approximation(mat *M, mat *U, mat *S, mat *V);

/* evaluate approximation to M using supplied partial pivoted QR decomposition */
void use_pivoted_QR_decomp_for_approximation(mat *M, mat *Qk, mat *Rk, vec *I);

/* evaluate approximation to M using supplied QB decomposition */
void use_QB_decomp_for_approximation(mat *M, mat *Q, mat *B);

/* evaluate approximation to M using supplied column ID of rank k */
void use_id_decomp_for_approximation(mat *M, mat *T, vec *I, int64_t k);

/* evaluate approximation to M using supplied two sided ID of rank k */
void use_id_two_sided_decomp_for_approximation(mat *M, mat *T, mat *S, vec *Icol, vec *Irow, int64_t k);
 
/* evaluate approximation to M using supplied CUR decomposition of rank k */
void use_cur_decomp_for_approximation(mat *M, mat *C, mat *U, mat *R);

