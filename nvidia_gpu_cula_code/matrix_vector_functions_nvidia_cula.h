/*
 * cula gpu code with host openmp 
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "omp.h"

#include "cula_lapack.h"


#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))


typedef struct {
    int nrows, ncols;
    double * d;
} mat;


typedef struct {
    int nrows;
    double * d;
} vec;


/* initialize new matrix and set all entries to zero */
mat * matrix_new(int nrows, int ncols);


/* initialize new vector and set all entries to zero */
vec * vector_new(int nrows);


void matrix_delete(mat *M);

void vector_delete(vec *v);


// column major format
void matrix_set_element(mat *M, int row_num, int col_num, double val);


double matrix_get_element(mat *M, int row_num, int col_num);


void vector_set_element(vec *v, int row_num, double val);


double vector_get_element(vec *v, int row_num);


void vector_set_data(vec *v, double *data);


/* scale vector by a constant */
void vector_scale(vec *v, double scalar);


/* compute euclidean norm of vector */
double vector_get2norm(vec *v);


/* copy contents of vec s to d  */
void vector_copy(vec *d, vec *s);


/* matrix access functions */
void matrix_get_col(mat *M, int j, vec *column_vec);

void matrix_set_col(mat *M, int j, vec *column_vec);

void matrix_get_row(mat *M, int i, vec *row_vec);

void matrix_set_row(mat *M, int i, vec *row_vec);


/* copy contents of mat S to D  */
void matrix_copy(mat *D, mat *S);


void matrix_copy_first_rows(mat *M_out, mat *M);

void matrix_copy_first_columns(mat *M_out, mat *M);



/* load matrix from file 
format:
% comment
num_rows num_columns num_nonzeros
row col nnz
.....
row col nnz
*/
mat * matrix_load_from_text_file(char *fname);


/* load matrix from binary file 
 * the nonzeros are in order of double loop over rows and columns
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
mat * matrix_load_from_binary_file(char *fname);

/* write matrix to binary file */
void matrix_write_to_binary_file(mat *M, char *fname);


/* load vector from file 
format:
% comment
num_rows
value
.....
value
*/
vec * vector_load_from_file(char *fname);


void matrix_print(mat * M);


void vector_print(vec * v);



/* keep only upper triangular matrix part as for symmetric matrix */
void matrix_copy_symmetric(mat *S, mat *M);



/* initialize diagonal matrix from vector data */
void initialize_diagonal_matrix(mat *D, vec *data);


/* invert diagonal matrix */
void invert_diagonal_matrix(mat *Dinv, mat *D);



/* initialize a random matrix */
void initialize_random_matrix(mat *M);


/* matrix frobenius norm */
double get_matrix_frobenius_norm(mat *M);


/* C = A*B ; column major */
void matrix_matrix_mult(mat *A, mat *B, mat *C);


/* C = A^T*B ; column major */
void matrix_transpose_matrix_mult(mat *A, mat *B, mat *C);


/* C = A*B^T ; column major */
void matrix_matrix_transpose_mult(mat *A, mat *B, mat *C);


/* y = M*x */
void matrix_vector_mult(mat *M, vec *x, vec *y);


/* y = M^T*x */
void matrix_transpose_vector_mult(mat *M, vec *x, vec *y);


/* set column of matrix to vector */
void matrix_set_col(mat *M, int j, vec *column_vec);


/* extract column of a matrix into a vector */
void matrix_get_col(mat *M, int j, vec *column_vec);



/* subtract b from a and save result in a  */
void vector_sub(vec *a, vec *b);



/* returns the dot product of two vectors */
double vector_dot_product(vec *u, vec *v);


/* subtract B from A and save result in A  */
void matrix_sub(mat *A, mat *B);



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



/* Performs [Q,R] = qr(M,'0') compact QR factorization 
M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */ 
void compact_QR_factorization(mat *M, mat *Q, mat *R);



/* returns Q from [Q,R] = qr(M,'0') compact QR factorization 
M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */ 
void QR_factorization_getQ(mat *M, mat *Q);



/* compute evals and evecs of symmetric matrix M
   matrix S must be symmetric
*/
void compute_evals_and_evecs_of_symm_matrix(mat *S, vec *evals);



/* computes SVD: M = U*S*V^T; note Vt is V transposed */
void singular_value_decomposition(mat *M, mat *U, mat *S, mat *Vt);



/* P = U*S*V^T */
void form_svd_product_matrix(mat *U, mat *S, mat *V, mat *P);


void append_matrices_horizontally(mat *A, mat *B, mat *C);


void append_matrices_vertically(mat *A, mat *B, mat *C);


/* calculate percent error between A and B: 100*norm(A - B)/norm(A) */
double get_percent_error_between_two_mats(mat *A, mat *B);


void checkStatus(culaStatus status);


/* for autorank 1 */
void estimate_rank_and_buildQ(mat *M, double frac_of_max_rank, double TOL, mat **Q, int *good_rank);

/* for autorank 2 */
void estimate_rank_and_buildQ2(mat *M, int kblock, double TOL, mat **Y, mat **Q, int *good_rank);

