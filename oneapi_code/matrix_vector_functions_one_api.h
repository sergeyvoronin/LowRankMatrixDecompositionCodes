#include <stdio.h>
#include "mkl.h"
#include "mkl_lapacke.h"
#include "mkl_vsl.h"
#include <stdint.h>
#include <math.h>
#include <inttypes.h>

#include <time.h>
#include <sys/time.h> // for clock_gettime()


#define SEED    777
#define BRNG    VSL_BRNG_MCG31
#define METHOD  VSL_RNG_METHOD_GAUSSIAN_ICDF

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))
// typedef MKL_INT64 myint64;
typedef int64_t myint64;


typedef struct {
    myint64 nrows, ncols;
    float * d;
} mat;


typedef struct {
    myint64 nrows;
    float * d;
} vec;


/* initialize new matrix and set all entries to zero */
mat * matrix_new(myint64 nrows, myint64 ncols);

/* initialize new vector and set all entries to zero */
vec * vector_new(myint64 nrows);

/* free space */
void matrix_delete(mat *M);
void vector_delete(vec *v);


/* set element in column major format */
void matrix_set_element(mat *M, myint64 row_num, myint64 col_num, float val);

/* get element in column major format */
float matrix_get_element(mat *M, myint64 row_num, myint64 col_num);


/* set vector element */
void vector_set_element(vec *v, myint64 row_num, float val);


/* get vector element */
float vector_get_element(vec *v, myint64 row_num);

/* load matrix from binary file */
mat * matrix_load_from_binary_file(char *fname);

/* write matrix to binary file */
void matrix_write_to_binary_file(mat *M, char *fname);


/* prmyint64 to terminal */
void matrix_print(mat * M);


/* prmyint64 to terminal */
void vector_print(vec * v);

/* v(:) = data */
void vector_set_data(vec *v, float *data);


/* scale vector by a constant */
void vector_scale(vec *v, float scalar);


/* scale matrix by a constant */
void matrix_scale(mat *M, float scalar);


/* compute euclidean norm of vector */
float vector_get2norm(vec *v);

/* min value and index of vector */
void vector_get_min_element(vec *v, myint64 *minindex, float *minval);

/* max value and index of vector */
void vector_get_max_element(vec *v, myint64 *maxindex, float *maxval);

/* copy contents of vec s to d  */
void vector_copy(vec *d, vec *s);


/* copy contents of mat S to D  */
void matrix_copy(mat *D, mat *S);


/* hard threshold matrix entries  */
void matrix_hard_threshold(mat *M, float TOL);


/* build transpose of matrix : Mt = M^T */
void matrix_build_transpose(mat *Mt, mat *M);


/* subtract b from a and save result in a  */
void vector_sub(vec *a, vec *b);


/* subtract B from A and save result in A  */
void matrix_sub(mat *A, mat *B);

/* A = A - u*v */
void matrix_sub_column_times_row_vector(mat *A, vec *u, vec *v);


/* matrix frobenius norm */
float get_matrix_frobenius_norm(mat *M);


/* matrix max abs val */
float get_matrix_max_abs_element(mat *M);

/* print out matrix */
void matrix_print(mat * M);

/* print out vector */
void vector_print(vec * v);


/* initialize random matrix (every elements follows Gaussian distribution) */
void initialize_random_matrix(mat *M);


/* C = A*B ; column major */
void matrix_matrix_mult(mat *A, mat *B, mat *C);


/* C = A^T*B ; column major */
void matrix_transpose_matrix_mult(mat *A, mat *B, mat *C);


/* C = A*B^T ; column major */
void matrix_matrix_transpose_mult(mat *A, mat *B, mat *C);


/* y = M*x ; column major */
void matrix_vector_mult(mat *M, vec *x, vec *y);


/* y = M^T*x ; column major */
void matrix_transpose_vector_mult(mat *M, vec *x, vec *y);



/* extract column of a matrix into a vector */
void matrix_get_col(mat *M, myint64 j, vec *column_vec);

/* set column of matrix to vector */
void matrix_set_col(mat *M, myint64 j, vec *column_vec);


/* extract row i of a matrix into a vector */
void matrix_get_row(mat *M, myint64 i, vec *row_vec);


/* put vector row_vec as row i of a matrix */
void matrix_set_row(mat *M, myint64 i, vec *row_vec);


/* Mc = M(:,inds) */
void matrix_get_selected_columns(mat *M, myint64 *inds, mat *Mc);


/* M(:,inds) = Mc */
void matrix_set_selected_columns(mat *M, myint64 *inds, mat *Mc);


/* Mr = M(inds,:) */
void matrix_get_selected_rows(mat *M, myint64 *inds, mat *Mr);


/* M(inds,:) = Mr */
void matrix_set_selected_rows(mat *M, myint64 *inds, mat *Mr);




/* copy only upper triangular matrix part of matrix in S from M */
void matrix_copy_symmetric(mat *S, mat *M);


/* keep only upper triangular matrix part of matrix M */
void matrix_keep_only_upper_triangular(mat *M);


/* initialize diagonal matrix from vector data */
void initialize_diagonal_matrix(mat *D, vec *data);



/* initialize identity */
void initialize_identity_matrix(mat *D);



/* invert diagonal matrix */
void invert_diagonal_matrix(mat *Dinv, mat *D);


/* overwrites supplied upper triangular matrix by its inverse */
void invert_upper_triangular_matrix(mat *Minv);


/* returns the dot product of two vectors */
float vector_dot_product(vec *u, vec *v);



/*
% project v in direction of u
function p=project_vec(v,u)
p = (dot(v,u)/norm(u)^2)*u;
*/
void project_vector(vec *v, vec *u, vec *p);


/* build orthonormal basis matrix
Q = Y;
for j=1:k
    vj = Q(:,j);
    for i=1:(j-1)
        vi = Q(:,i);
        vj = vj - project_vec(vj,vi);
    end
    vj = vj/norm(vj);
    Q(:,j) = vj;
end
*/
void build_orthonormal_basis_from_mat(mat *A, mat *Q);



void fill_vector_from_row_list(vec *input, vec *inds, vec *output);


void matrix_copy_first_rows(mat *M_out, mat *M);


void matrix_copy_first_columns(mat *M_out, mat *M);

void matrix_copy_first_columns_with_param(mat *D, mat *S, myint64 num_columns);


void matrix_copy_first_k_rows_and_columns(mat *M_out, mat *M);

void matrix_copy_all_rows_and_last_columns_from_indexk(mat *M_out, mat *M, myint64 k);


void fill_matrix_from_first_rows(mat *M, myint64 k, mat *M_k);

void fill_matrix_from_last_rows(mat *M, myint64 k, mat *M_k);


void fill_matrix_from_first_columns(mat *M, myint64 k, mat *M_k);

void fill_matrix_from_last_columns(mat *M, myint64 k, mat *M_k);

void fill_matrix_from_last_columns_from_specified_one(mat *M, myint64 k, mat *M_k);


void fill_matrix_from_lower_right_corner(mat *M, myint64 k, mat *M_out);


/* M_k = M(:,I(1:k)) */
void fill_matrix_from_first_columns_from_list(mat *M, vec *I, myint64 k, mat *M_k);

/* M_k = M(I(1:k),:) */
void fill_matrix_from_first_rows_from_list(mat *M, vec *I, myint64 k, mat *M_k);
 

void fill_matrix_from_last_columns_from_list(mat *M, vec *I, myint64 k, mat *M_k);


/* M = M(:,1:k); */
void resize_matrix_by_columns(mat **M, myint64 k);

/* M = M(:,(end-k):end); */
void resize_matrix_by_columns_from_end(mat **M, myint64 k);

/* M = M(1:k,:); */
void resize_matrix_by_rows(mat **M, myint64 k);

/* M = M((end-k):end,:); */
void resize_matrix_by_rows_from_end(mat **M, myint64 k);


//void fill_matrix_from_column_list(mat *M, vec *I, mat *M_k);
//void fill_matrix_from_row_list(mat *M, vec *I, mat *M_k);


void append_matrices_horizontally(mat *A, mat *B, mat *C);

 
void append_matrices_vertically(mat *A, mat *B, mat *C);

/* Iinv(I)=[0:length(I)-1] */
void vector_build_rewrapped(vec *Iinv, vec *I);


/* calculate percent error between A and B: 100*norm(A - B)/norm(A) */
float get_percent_error_between_two_mats(mat *A, mat *B);


float get_matrix_column_norm_squared(mat *M, myint64 colnum);


float matrix_getmaxcolnorm(mat *M);


void compute_matrix_column_norms(mat *M, vec *column_norms);



/* compute eigendecomposition of symmetric matrix M
*/
void compute_evals_and_evecs_of_symm_matrix(mat *S, vec *evals);


void singular_value_decomposition(mat *M, mat *U, mat *S, mat *Vt);

/* Performs [Q,R] = qr(M,'0') compact QR factorization 
M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */ 
void compact_QR_factorization(mat *M, mat *Q, mat *R);


/* returns Q from [Q,R] = qr(M,'0') compact QR factorization 
M is mxn ; Q is mxn ; R is not computed */ 
void QR_factorization_getQ(mat *M, mat *Q);


/* for autorank 1 */
void estimate_rank_and_buildQ(mat *M, float frac_of_max_rank, float TOL, mat **Q, myint64 *good_rank);

/* for autorank 2 */
void estimate_rank_and_buildQ2(mat *M, myint64 kblock, float TOL, mat **Y, mat **Q, myint64 *good_rank);

/* P = U * S * Vt */
void form_svd_product_matrix(mat *U, mat *S, mat *V, mat *P);

/* P = C * U * R */
void form_cur_product_matrix(mat *C, mat *U, mat *R, mat *P);

/* solve A X = B with A upper triangular */
void upper_triangular_system_solve(mat *A, mat *B, mat *X, myint64 solve_type);

/* solve A X = B with A square matrix */
void square_matrix_system_solve(mat *A, mat *X, mat *B);

/* get seconds for recording runtime */
//float get_seconds_frac(struct timeval start_timeval, struct timeval end_timeval);
float get_seconds_frac(float start_timeval, float end_timeval);

double omp_get_wtime(void);
