/*
 * gpu device based rsvd code, uses cuda r18 for blas/lapack
 * Sergey Voronin, 2015 - 2016
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "cula_lapack.h"

#include <time.h>
#include <sys/time.h> // for clock_gettime()
#include <cuda_runtime.h>



#define min(x,y) (((x) < (y)) ? (x) : (y))


typedef struct {
    int nrows, ncols;
    double * d;
    double * ddev;
} mat;



typedef struct {
    int nrows;
    double * d;
    double * ddev;
} vec;


/* initialize new matrix and set all entries to zero */
mat * matrix_new(int nrows, int ncols)
{
    mat *M = malloc(sizeof(mat));
    M->d = (double*)calloc(nrows*ncols, sizeof(double));
    cudaMalloc((void **)&(M->ddev), nrows*ncols*sizeof(double));
    cudaMemset(M->ddev, 0, nrows*ncols);
    M->nrows = nrows;
    M->ncols = ncols;
    return M;
}


/* initialize new vector and set all entries to zero */
vec * vector_new(int nrows)
{
    vec *v = malloc(sizeof(vec));
    v->d = (double*)calloc(nrows,sizeof(double));
    cudaMalloc((void **)&(v->ddev), nrows*sizeof(double));
    cudaMemset(v->ddev, 0, nrows);
    v->nrows = nrows;
    return v;
}

void matrix_elems_copy_to_device(mat *M){
    cudaMemcpy( M->ddev, M->d, (M->nrows)*(M->ncols)*sizeof(double), cudaMemcpyHostToDevice);
}

void matrix_elems_copy_from_device(mat *M){
    cudaMemcpy( M->d, M->ddev, (M->nrows)*(M->ncols)*sizeof(double), cudaMemcpyDeviceToHost);
}




void vector_elems_copy_to_device(vec *v){
    cudaMemcpy( v->ddev, v->d, (v->nrows)*sizeof(double), cudaMemcpyHostToDevice);
}


void vector_elems_copy_from_device(vec *v){
    cudaMemcpy( v->d, v->ddev, (v->nrows)*sizeof(double), cudaMemcpyDeviceToHost);
}



void matrix_delete(mat *M)
{
    free(M->d);
    cudaFree(M->ddev);
    free(M);
}


void vector_delete(vec *v)
{
    free(v->d);
    cudaFree(v->ddev);
    free(v);
}


// column major format
void matrix_set_element(mat *M, int row_num, int col_num, double val){
    //M->d[row_num*(M->ncols) + col_num] = val;
    M->d[col_num*(M->nrows) + row_num] = val;
}



double matrix_get_element(mat *M, int row_num, int col_num){
    //return M->d[row_num*(M->ncols) + col_num];
    return M->d[col_num*(M->nrows) + row_num];
}


void vector_set_element(vec *v, int row_num, double val){
    v->d[row_num] = val;
}


double vector_get_element(vec *v, int row_num){
    return v->d[row_num];
}


void vector_set_data(vec *v, double *data){
    int i;
    for(i=0; i<(v->nrows); i++){
        v->d[i] = data[i];
    }
}


/* scale vector by a constant */
void vector_scale(vec *v, double scalar){
    int i;
    for(i=0; i<(v->nrows); i++){
        v->d[i] = scalar*(v->d[i]);
    }
}


/* compute euclidean norm of vector */
double vector_get2norm(vec *v){
    int i;
    double val, normval = 0;
    for(i=0; i<(v->nrows); i++){
        val = v->d[i];
        normval += val*val;
    }
    return sqrt(normval);
}



/* copy contents of vec s to d  */
void vector_copy(vec *d, vec *s){
    int i;
    for(i=0; i<(s->nrows); i++){
        d->d[i] = s->d[i];
    }
}


/* copy contents of mat S to D  */
void matrix_copy(mat *D, mat *S){
    int i;
    matrix_elems_copy_from_device(S);
    for(i=0; i<((S->nrows)*(S->ncols)); i++){
        D->d[i] = S->d[i];
    }
    matrix_elems_copy_to_device(D);
}


/* load matrix from file 
format:
% comment
num_rows num_columns num_nonzeros
row col nnz
.....
row col nnz
*/
mat * matrix_load_from_text_file(char *fname){
    int i, j, num_rows, num_columns, num_nonzeros, row_num, col_num;
    double nnz_val;
    char *nnz_val_str;
    char *line;
    FILE *fp;
    mat *M;
    
    line = (char*)malloc(200*sizeof(char));
    fp = fopen(fname,"r");
    fgets(line,100,fp); //read comment
    fgets(line,100,fp); //read dimensions and nnzs 
    sscanf(line, "%d %d %d", &num_rows, &num_columns, &num_nonzeros);
    M = matrix_new(num_rows,num_columns);

    // read and set elements
    nnz_val_str = (char*)malloc(50*sizeof(char));
    for(i=0; i<(num_nonzeros); i++){
        fgets(line,100,fp); 
        sscanf(line, "%d %d %s", &row_num, &col_num, nnz_val_str);
        nnz_val = atof(nnz_val_str);
        matrix_set_element(M,row_num,col_num,nnz_val);
    }
    fclose(fp);

    // clean
    free(line);
    free(nnz_val_str);

    return M;
}



/* load matrix from binary file 
 * the nonzeros are in order of double loop over rows and columns
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
mat * matrix_load_from_binary_file(char *fname){
    int i, j, num_rows, num_columns, row_num, col_num;
    double nnz_val;
    size_t one = 1;
    FILE *fp;
    mat *M;
    
    fp = fopen(fname,"r");
    fread(&num_rows,sizeof(int),one,fp); //read m
    fread(&num_columns,sizeof(int),one,fp); //read n
    printf("initializing M of size %d by %d\n", num_rows, num_columns);
    M = matrix_new(num_rows,num_columns);
    printf("done..\n");

    // read and set elements
    for(i=0; i<num_rows; i++){
        for(j=0; j<num_columns; j++){
            fread(&nnz_val,sizeof(double),one,fp); //read nnz
            matrix_set_element(M,i,j,nnz_val);
        }
    }
    fclose(fp);

    matrix_elems_copy_to_device(M);

    return M;
}




/* load vector from file 
format:
% comment
num_rows
value
.....
value
*/
vec * vector_load_from_file(char *fname){
    int i, j, num_rows;
    double nnz_val;
    char *nnz_val_str;
    char *line;
    FILE *fp;
    vec *v;

    line = (char*)malloc(200*sizeof(char));
    fp = fopen(fname,"r");
    fgets(line,100,fp); //read comment
    fgets(line,100,fp); //read dimension 
    sscanf(line, "%d", &num_rows);
    v = vector_new(num_rows);

    // read and set elements
    nnz_val_str = (char*)malloc(50*sizeof(char));
    for(i=0; i<num_rows; i++){
        fgets(line,100,fp);
        sscanf(line, "%s", nnz_val_str);
        nnz_val = atof(nnz_val_str);
        vector_set_element(v, i, nnz_val);
    }
    fclose(fp);

    // clean
    free(line);
    free(nnz_val_str);

    return v;
}



void matrix_print(mat * M){
    int i,j;
    double val;
    for(i=0; i<M->nrows; i++){
        for(j=0; j<M->ncols; j++){
            val = matrix_get_element(M, i, j);
            printf("%f  ", val);
        }
        printf("\n");
    }
}


void vector_print(vec * v){
    int i;
    double val;
    for(i=0; i<v->nrows; i++){
        val = vector_get_element(v, i);
        printf("%f\n", val);
    }
}



/* keep only upper triangular matrix part as for symmetric matrix */
void matrix_copy_symmetric(mat *S, mat *M){
    int i,j,n,m;
    m = M->nrows;
    n = M->ncols;
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            if(j>=i){
                matrix_set_element(S,i,j,matrix_get_element(M,i,j));
            }
        }
    }
}


/* initialize diagonal matrix from vector data */
void initialize_diagonal_matrix(mat *D, vec *data){
    int i;
    vector_elems_copy_from_device(data);
    for(i=0; i<(D->nrows); i++){
        matrix_set_element(D,i,i,data->d[i]);
    }
    matrix_elems_copy_to_device(D);
}


/* invert diagonal matrix */
void invert_diagonal_matrix(mat *Dinv, mat *D){
    int i;
    matrix_elems_copy_from_device(D);
    for(i=0; i<(D->nrows); i++){
        matrix_set_element(Dinv,i,i,1.0/(matrix_get_element(D,i,i)));
    }
    matrix_elems_copy_to_device(Dinv);
}



/* initialize a random matrix */
void initialize_random_matrix(mat *M){
    int i,m,n;
    double val;
    m = M->nrows;
    n = M->ncols;

    // seed 
    srand(time(NULL));

    // read and set elements
    for(i=0; i<(m*n); i++){
        val = ((double) rand() / (RAND_MAX));
        M->d[i] = val;
    }
    matrix_elems_copy_to_device(M);
}



/* matrix frobenius norm */
double matrix_frobenius_norm(mat *M){
    int i;
    double val, normval = 0;
    matrix_elems_copy_from_device(M);
    for(i=0; i<((M->nrows)*(M->ncols)); i++){
        val = M->d[i];
        normval += val*val;
    }
    return sqrt(normval);
}


/* C = A*B ; column major */
void matrix_matrix_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A->nrows, B->ncols, A->ncols, alpha, A->d, A->ncols, B->d, B->ncols, beta, C->d, C->ncols);
    culaDeviceDgemm('N', 'N', A->nrows, B->ncols, A->ncols, alpha, A->ddev, A->nrows, B->ddev, B->nrows, beta, C->ddev, C->nrows);
}


/* C = A^T*B ; column major */
void matrix_transpose_matrix_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    //cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A->ncols, B->ncols, A->nrows, alpha, A->d, A->ncols, B->d, B->ncols, beta, C->d, C->ncols);
    culaDeviceDgemm('T', 'N', A->ncols, B->ncols, A->nrows, alpha, A->ddev, A->nrows, B->ddev, B->nrows, beta, C->ddev, C->nrows);
}


/* C = A*B^T ; column major */
void matrix_matrix_transpose_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, A->nrows, B->nrows, A->ncols, alpha, A->d, A->ncols, B->d, B->ncols, beta, C->d, C->ncols);
    culaDeviceDgemm('N', 'T', A->nrows, B->nrows, A->ncols, alpha, A->ddev, A->nrows, B->ddev, B->nrows, beta, C->ddev, C->nrows);
}


/* y = M*x */
void matrix_vector_mult(mat *M, vec *x, vec *y){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    //cblas_dgemv (CblasRowMajor, CblasNoTrans, M->nrows, M->ncols, alpha, M->d, M->ncols, x->d, 1, beta, y->d, 1);
    culaDeviceDgemv ('N', M->nrows, M->ncols, alpha, M->ddev, M->nrows, x->ddev, 1, beta, y->ddev, 1);
}


/* y = M^T*x */
void matrix_transpose_vector_mult(mat *M, vec *x, vec *y){
    //gsl_blas_dgemv (CblasNoTrans, 1.0, M, x, 0.0, y);
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    //cblas_dgemv (CblasRowMajor, CblasTrans, M->nrows, M->ncols, alpha, M->d, M->ncols, x->d, 1, beta, y->d, 1);
    culaDeviceDgemv ('T', M->nrows, M->ncols, alpha, M->ddev, M->nrows, x->ddev, 1, beta, y->ddev, 1);
}



/* set column of matrix to vector */
void matrix_set_col(mat *M, int j, vec *column_vec){
    int i;
    {
    for(i=0; i<M->nrows; i++){
        matrix_set_element(M,i,j,vector_get_element(column_vec,i));
    }
    }
}


/* extract column of a matrix into a vector */
void matrix_get_col(mat *M, int j, vec *column_vec){
    int i;
    {
    for(i=0; i<M->nrows; i++){ 
        vector_set_element(column_vec,i,matrix_get_element(M,i,j));
    }
    }
}


/* extract row i of a matrix into a vector */
void matrix_get_row(mat *M, int i, vec *row_vec){
    int j;
    {
    for(j=0; j<M->ncols; j++){ 
        vector_set_element(row_vec,j,matrix_get_element(M,i,j));
    }
    }
}


/* put vector row_vec as row i of a matrix */
void matrix_set_row(mat *M, int i, vec *row_vec){
    int j;
    {
    for(j=0; j<M->ncols; j++){ 
        matrix_set_element(M,i,j,vector_get_element(row_vec,j));
    }
    }
}



/* subtract b from a and save result in a  */
void vector_sub(vec *a, vec *b){
    int i;
    for(i=0; i<(a->nrows); i++){
        a->d[i] = a->d[i] - b->d[i];
    }
}



/* returns the dot product of two vectors */
double vector_dot_product(vec *u, vec *v){
    int i;
    double dotval = 0;
    for(i=0; i<u->nrows; i++){
        dotval += (u->d[i])*(v->d[i]);
    }
    return dotval;
}


/* subtract B from A and save result in A  */
void matrix_sub(mat *A, mat *B){
    int i;
    matrix_elems_copy_from_device(A);
    matrix_elems_copy_from_device(B);
    for(i=0; i<((A->nrows)*(A->ncols)); i++){
        A->d[i] = A->d[i] - B->d[i];
    }
    matrix_elems_copy_to_device(A);
}



/*
% project v in direction of u
function p=project_vec(v,u)
p = (dot(v,u)/norm(u)^2)*u;
*/
void project_vector(vec *v, vec *u, vec *p){
    double dot_product_val, vec_norm, scalar_val; 
    dot_product_val = vector_dot_product(v, u);
    vec_norm = vector_get2norm(u);
    scalar_val = dot_product_val/(vec_norm*vec_norm);
    vector_copy(p, u);
    vector_scale(p, scalar_val); 
}


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
void build_orthonormal_basis_from_mat(mat *A, mat *Q){
    int m,n,i,j,ind,num_ortos=2;
    double vec_norm;
    vec *vi,*vj,*p;
    m = A->nrows;
    n = A->ncols;
    vi = vector_new(m);
    vj = vector_new(m);
    p = vector_new(m);
    matrix_copy(Q, A);

    for(ind=0; ind<num_ortos; ind++){
        for(j=0; j<n; j++){
            matrix_get_col(Q,j,vj);
            for(i=0; i<j; i++){
                matrix_get_col(Q, i, vi);
                project_vector(vj, vi, p);
                vector_sub(vj, p);
            }
            vec_norm = vector_get2norm(vj);
            vector_scale(vj, 1.0/vec_norm);
            matrix_set_col(Q, j, vj);
        }
    }
}



/* Performs [Q,R] = qr(M,'0') compact QR factorization 
M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */ 
void compact_QR_factorization(mat *M, mat *Q, mat *R){
    int i,j,m,n,k;
    m = M->nrows; n = M->ncols;
    k = min(m,n);
    mat *R_full = matrix_new(m,n);
    matrix_copy(R_full,M);
    vec *tau = vector_new(m);

    // get R
    //LAPACKE_dgeqrf(CblasRowMajor, m, n, R_full->d, n, tau->d);
    culaDeviceDgeqrf(m, n, R_full->ddev, m, tau->ddev);
    matrix_elems_copy_from_device(R_full);
    
    for(i=0; i<k; i++){
        for(j=0; j<k; j++){
            if(j>=i){
                matrix_set_element(R,i,j,matrix_get_element(R_full,i,j));
            }
        }
    }

    matrix_elems_copy_to_device(R);
    matrix_elems_copy_to_device(R_full);

    // get Q
    matrix_copy(Q,R_full); 
    //LAPACKE_dorgqr(CblasRowMajor, m, n, n, Q->d, n, tau->d);
    culaDeviceDorgqr(m, n, n, Q->ddev, m, tau->ddev);

    // clean up
    matrix_delete(R_full);
    vector_delete(tau);
}




/* returns Q from [Q,R] = qr(M,'0') compact QR factorization 
M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */ 
void QR_factorization_getQ(mat *M, mat *Q){
    int i,j,m,n,k;
    m = M->nrows; n = M->ncols;
    k = min(m,n);
    matrix_copy(Q,M);
    vec *tau = vector_new(m);

    culaDeviceDgeqrf(m, n, Q->ddev, m, tau->ddev);
    culaDeviceDorgqr(m, n, n, Q->ddev, m, tau->ddev);

    // clean up
    vector_delete(tau);
}



/* compute evals and evecs of symmetric matrix M
   matrix S must be symmetric
*/
void compute_evals_and_evecs_of_symm_matrix(mat *S, vec *evals){
    //LAPACKE_dsyev( LAPACK_ROW_MAJOR, 'V', 'U', S->nrows, S->d, S->nrows, evals->d);
    culaDeviceDsyev('V', 'U', S->nrows, S->ddev, S->ncols, evals->ddev);
}



/* computes SVD: M = U*S*V^T; note Vt is V transposed */
void singular_value_decomposition(mat *M, mat *U, mat *S, mat *Vt){
    int m,n,k;
    m = M->nrows; n = M->ncols;
    k = min(m,n);
    vec * work = vector_new(k);
    vec * svals = vector_new(k);
    //LAPACKE_dgesvd( LAPACK_ROW_MAJOR, 'A', 'A', m, n, M->d, n, svals->d, U->d, m, Vt->d, n, work->d );
    culaDeviceDgesvd('A', 'A', k, k, M->ddev, k, svals->ddev, U->ddev, k, Vt->ddev, k);

    vector_elems_copy_from_device(svals);
    initialize_diagonal_matrix(S, svals);
    vector_delete(work);
    vector_delete(svals);
}


/* M_k = M(1:k,:) */
void fill_matrix_from_first_rows(mat *M, int k, mat *M_k){
    int i;
    vec *row_vec;
    matrix_elems_copy_from_device(M);
    {
    for(i=0; i<k; i++){
        row_vec = vector_new(M->ncols);
        matrix_get_row(M,i,row_vec);
        matrix_set_row(M_k,i,row_vec);
        vector_delete(row_vec);
    }
    }
    matrix_elems_copy_to_device(M_k);
}


/* M_k = M((end-k):end,:) */
void fill_matrix_from_last_rows(mat *M, int k, mat *M_k){
    int i;
    vec *row_vec;
    matrix_elems_copy_from_device(M);
    {
    for(i=0; i<k; i++){
        row_vec = vector_new(M->nrows);
        matrix_get_row(M,M->nrows - k +i,row_vec);
        matrix_set_row(M_k,i,row_vec);
        vector_delete(row_vec);
    }
    }
    matrix_elems_copy_to_device(M_k);
}


void fill_matrix_from_first_columns(mat *M, int k, mat *M_k){
    int i;
    vec *col_vec;
    matrix_elems_copy_from_device(M);
    {
    for(i=0; i<k; i++){
        col_vec = vector_new(M->nrows);
        matrix_get_col(M,i,col_vec);
        matrix_set_col(M_k,i,col_vec);
        vector_delete(col_vec);
    }
    }
    matrix_elems_copy_to_device(M_k);
}


/* M_k = M(:,(end-k:end) */
void fill_matrix_from_last_columns(mat *M, int k, mat *M_k){
    int i;
    vec *col_vec;
    matrix_elems_copy_from_device(M);
    {
    for(i=0; i<k; i++){
        col_vec = vector_new(M->nrows);
        matrix_get_col(M,M->ncols - k +i,col_vec);
        matrix_set_col(M_k,i,col_vec);
        vector_delete(col_vec);
    }
    }
    matrix_elems_copy_to_device(M_k);
}


/* M = M(:,1:k); */
void resize_matrix_by_columns(mat **M, int k){
    int j;
    mat *R;
    R = matrix_new((*M)->nrows, k);
    fill_matrix_from_first_columns(*M, k, R);
    matrix_delete(*M);
    *M = matrix_new(R->nrows, R->ncols);
    matrix_copy(*M,R);
    matrix_delete(R);
}  


/* M = M(:,(end-k+1):end); */
void resize_matrix_by_columns_from_end(mat **M, int k){
    int j;
    mat *R;
    R = matrix_new((*M)->nrows, k);
    fill_matrix_from_last_columns(*M, k, R);
    matrix_delete(*M);
    *M = matrix_new(R->nrows, R->ncols);
    matrix_copy(*M,R);
    matrix_delete(R);
}  



/* M = M(1:k,:); */
void resize_matrix_by_rows(mat **M, int k){
    int j;
    mat *R;
    R = matrix_new(k, (*M)->ncols);
    fill_matrix_from_first_rows(*M, k, R);
    matrix_delete(*M);
    *M = matrix_new(R->nrows, R->ncols);
    matrix_copy(*M,R);
    matrix_delete(R);
}


/* M = M((end-k+1):end,:); */
void resize_matrix_by_rows_from_end(mat **M, int k){
    int j;
    mat *R;
    R = matrix_new(k, (*M)->ncols);
    fill_matrix_from_last_rows(*M, k, R);
    matrix_delete(*M);
    *M = matrix_new(R->nrows, R->ncols);
    matrix_copy(*M,R);
    matrix_delete(R);
}




/* computes the approximate low rank SVD of rank k of matrix M using BBt version */
void randomized_low_rank_svd1(mat *M, int k, mat *U, mat *S, mat *V){
    int i,j,m,n;
    double val;
    m = M->nrows; n = M->ncols;

    // build random matrix
    mat *RN = matrix_new(n, k);
    initialize_random_matrix(RN);

    // multiply to get matrix of random samples Y
    printf("form Y..\n");
    mat *Y = matrix_new(m,k);
    matrix_matrix_mult(M, RN, Y);

    // build Q from Y
    printf("form Q..\n");
    mat *Q = matrix_new(m,k);
    //build_orthonormal_basis_from_mat(Y,Q);
    QR_factorization_getQ(Y, Q);


    // build the matrix B B^T = Q^T M M^T Q column by column 
    // Bt = M^T Q ; nxm * mxk = nxk
    printf("form BBt..\n");
    mat *B = matrix_new(k,n);
    matrix_transpose_matrix_mult(Q,M,B);

    mat *Bt = matrix_new(n,k);
    matrix_transpose_matrix_mult(M,Q,Bt);    

    mat *BBt = matrix_new(k,k);
    matrix_matrix_mult(B,Bt,BBt);    

    // compute eigendecomposition of BBt
    printf("eigendecompose BBt..\n");
    vec *evals = vector_new(k);
    mat *Uhat = matrix_new(k, k);
    matrix_copy_symmetric(Uhat,BBt);
    compute_evals_and_evecs_of_symm_matrix(Uhat, evals);


    // compute singular values and matrix Sigma
    printf("form S..\n");
    vec *singvals = vector_new(k);
    for(i=0; i<k; i++){
        vector_set_element(singvals,i,sqrt(vector_get_element(evals,i)));
    }
    initialize_diagonal_matrix(S, singvals);
    
    // compute U = Q*Uhat mxk * kxk = mxk  
    printf("form U..\n");
    matrix_matrix_mult(Q,Uhat,U);

    // compute nxk V 
    // V = B^T Uhat * Sigma^{-1}
    printf("form V..\n");
    mat *Sinv = matrix_new(k,k);
    mat *UhatSinv = matrix_new(k,k);
    invert_diagonal_matrix(Sinv,S);
    matrix_matrix_mult(Uhat,Sinv,UhatSinv);
    matrix_matrix_mult(Bt,UhatSinv,V);

    // clean up
    matrix_delete(RN);
    matrix_delete(Y);
    matrix_delete(Q);
    matrix_delete(B);
    matrix_delete(Bt);
    matrix_delete(Sinv);
    matrix_delete(UhatSinv);
}



/* computes the approximate low rank SVD of rank k of matrix M using QR version */
void randomized_low_rank_svd2(mat *M, int k, mat *U, mat *S, mat *V){
    int i,j,m,n;
    double val;
    m = M->nrows; n = M->ncols;

    // build random matrix
    mat *RN = matrix_new(n, k);
    initialize_random_matrix(RN);

    // multiply to get matrix of random samples Y
    printf("form Y..\n");
    mat *Y = matrix_new(m,k);
    matrix_matrix_mult(M, RN, Y);

    // build Q from Y
    printf("form Q..\n");
    mat *Q = matrix_new(m,k);
    //build_orthonormal_basis_from_mat(Y,Q);
    QR_factorization_getQ(Y, Q);

    // form Bt = Mt*Q : nxm * mxk = nxk
    printf("form Bt..\n");
    mat *Bt = matrix_new(n,k);
    matrix_transpose_matrix_mult(M,Q,Bt);

    // compute QR factorization of Bt    
    //M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */ 
    //void compact_QR_factorization(mat *M, mat *Q, mat *R)
    printf("doing QR..\n");
    mat *Qhat = matrix_new(n,k);
    mat *Rhat = matrix_new(k,k);   
    compact_QR_factorization(Bt,Qhat,Rhat);

    // compute SVD of Rhat (kxk)
    printf("doing SVD..\n");
    mat *Uhat = matrix_new(k,k);
    mat *Vhat_trans = matrix_new(k,k);
    singular_value_decomposition(Rhat, Uhat, S, Vhat_trans);

    // U = Q*Vhat_trans
    printf("form U..\n");
    matrix_matrix_transpose_mult(Q,Vhat_trans,U);

    // V = Qhat*Uhat
    printf("form V..\n");
    matrix_matrix_mult(Qhat,Uhat,V);

    // free stuff
    matrix_delete(RN);
    matrix_delete(Y);
    matrix_delete(Q);
    matrix_delete(Rhat);
    matrix_delete(Qhat);
    matrix_delete(Uhat);
    matrix_delete(Vhat_trans);
    matrix_delete(Bt);
}


/* computes the approximate low rank SVD of rank k or tolerance TOL of matrix M  */
void low_rank_svd_rand_decomp_fixed_rank(mat *M, int k, int p, int vnum, int q, int s, int *frank, mat **U, mat **S, mat **V){
    int i,j,m,n,r,l;
    mat *RN, *Y, *Z, *Q, *Yorth, *Zorth;
    mat *Bt, *Qhat, *Rhat, *Uhat, *Vhat_trans;
    
    // get dims
    m = M->nrows;
    n = M->ncols; 
    r = min(m,n);

    // setup mats
    l = k + p;
    *U = matrix_new(m,l);
    *S = matrix_new(l,l);
    *V = matrix_new(n,l);

    // build random matrix
    RN = matrix_new(n, l);
    initialize_random_matrix(RN);

    // multiply to get matrix of random samples Y
    Y = matrix_new(m, l);
    matrix_matrix_mult(M, RN, Y);

    // now build up (M M^T)^q R
    Z = matrix_new(n,l);
    Yorth = matrix_new(m,l);
    Zorth = matrix_new(n,l);
    for(j=1; j<q; j++){
        printf("M M^T mult j=%d of %d\n", j, q-1);

        if((2*j-2) % s == 0){
            //printf("orthogonalize Y..\n");
            QR_factorization_getQ(Y, Yorth);
            //printf("Z = M'*Yorth..\n");
            matrix_transpose_matrix_mult(M,Yorth,Z);
        }
        else{
            //printf("Z = M'*Y..\n");
            matrix_transpose_matrix_mult(M,Y,Z);
        }

        
        if((2*j-1) % s == 0){
            //printf("orthogonalize Z..\n");
            QR_factorization_getQ(Z, Zorth);
            //printf("Y = M*Zorth..\n");
            matrix_matrix_mult(M,Zorth,Y);
        }
        else{
            //printf("Y = M*Z..\n");
            matrix_matrix_mult(M,Z,Y);
        }
    }

    // orthogonalize on exit from loop to get Q
    Q = matrix_new(m,l);
    QR_factorization_getQ(Y, Q);

    // either QR of B^T method, or eigendecompose BB^T method
    if(vnum == 1 || vnum > 2){
        printf("using QR of B^T method\n");
        
        // form Bt = Mt*Q : nxm * mxl = nxl
        //printf("form Bt..\n");
        Bt = matrix_new(n,l);
        matrix_transpose_matrix_mult(M,Q,Bt);

        // compute QR factorization of Bt    
        //M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */ 
        //printf("doing QR..\n");
        Qhat = matrix_new(n,l);
        Rhat = matrix_new(l,l);   
        compact_QR_factorization(Bt,Qhat,Rhat);

        // compute SVD of Rhat (lxl)
        //printf("doing SVD..\n");
        Uhat = matrix_new(l,l);
        Vhat_trans = matrix_new(l,l);
        singular_value_decomposition(Rhat, Uhat, *S, Vhat_trans);

        // U = Q*Vhat_trans
        //printf("form U..\n");
        matrix_matrix_transpose_mult(Q,Vhat_trans,*U);

        // V = Qhat*Uhat
        //printf("form V..\n");
        matrix_matrix_mult(Qhat,Uhat,*V);

        // clean up
        matrix_delete(Rhat);
        matrix_delete(Qhat);
        matrix_delete(Uhat);
        matrix_delete(Vhat_trans);
        matrix_delete(Bt);

        // resize matrices to rank k from beginning
        //printf("resize mats\n");
        resize_matrix_by_columns(U,k);
        resize_matrix_by_columns(V,k);
        resize_matrix_by_columns(S,k);
        resize_matrix_by_rows(S,k);
    } else {
        printf("using eigendecomposition of B B^T method\n");
        // build the matrix B B^T = Q^T M M^T Q column by column 
        // Bt = M^T Q ; nxm * mxk = nxk
        printf("form BBt..\n");
        mat *B = matrix_new(l,n);
        matrix_transpose_matrix_mult(Q,M,B);

        mat *BBt = matrix_new(l,l);
        matrix_matrix_transpose_mult(B,B,BBt);    

        // compute eigendecomposition of BBt
        printf("eigendecompose BBt..\n");
        vec *evals = vector_new(l);
        mat *Uhat = matrix_new(l,l);
        matrix_copy_symmetric(Uhat,BBt);
        compute_evals_and_evecs_of_symm_matrix(Uhat, evals);


        // compute singular values and matrix Sigma
        printf("form S..\n");
        vec *singvals = vector_new(l);
        for(i=0; i<l; i++){
            vector_set_element(singvals,i,sqrt(vector_get_element(evals,i)));
        }
        initialize_diagonal_matrix(*S, singvals);
        
        // compute U = Q*Uhat mxk * kxk = mxk  
        printf("form U..\n");
        matrix_matrix_mult(Q,Uhat,*U);

        // compute nxk V 
        // V = B^T Uhat * Sigma^{-1}
        printf("form V..\n");
        mat *Sinv = matrix_new(l,l);
        mat *UhatSinv = matrix_new(l,l);
        invert_diagonal_matrix(Sinv,*S);
        matrix_matrix_mult(Uhat,Sinv,UhatSinv);
        matrix_transpose_matrix_mult(B,UhatSinv,*V);

        matrix_delete(BBt);
        matrix_delete(Sinv);
        matrix_delete(UhatSinv);

        // resize matrices to rank k from end
        printf("resize mats to rank %d from end\n",k);
        resize_matrix_by_columns_from_end(U,k);
        resize_matrix_by_columns_from_end(V,k);
        resize_matrix_by_columns_from_end(S,k);
        resize_matrix_by_rows_from_end(S,k);
    }

    //printf("free stuff\n");
    matrix_delete(RN);
    matrix_delete(Y);
    matrix_delete(Q);
    matrix_delete(Z);
    matrix_delete(Yorth);
    matrix_delete(Zorth);
}


/* P = U*S*V^T */
void form_svd_product_matrix(mat *U, mat *S, mat *V, mat *P){
    int k,m,n;
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    m = P->nrows;
    n = P->ncols;
    k = S->nrows;
    mat * SVt = matrix_new(k,n);

    // form SVt = S*V^T
    matrix_matrix_transpose_mult(S,V,SVt);

    // form P = U*S*V^T
    matrix_matrix_mult(U,SVt,P);
}


/* calculate percent error between A and B: 100*norm(A - B)/norm(A) */
double get_percent_error_between_two_mats(mat *A, mat *B){
    int m,n;
    double normA, normB, normA_minus_B;
    m = A->nrows;
    n = A->ncols;
    mat *A_minus_B = matrix_new(m,n);
    matrix_copy(A_minus_B, A);
    matrix_sub(A_minus_B, B);
    normA = matrix_frobenius_norm(A);
    normB = matrix_frobenius_norm(B);
    normA_minus_B = matrix_frobenius_norm(A_minus_B);
    return 100.0*normA_minus_B/normA;
}


void checkStatus(culaStatus status)
{
    char buf[256];

    if(!status)
        return;

    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    printf("%s\n", buf);

    culaShutdown();
    exit(EXIT_FAILURE);
}


double get_seconds_frac(struct timeval start_timeval, struct timeval end_timeval){
    long secs_used, micros_used;
    secs_used=(end_timeval.tv_sec - start_timeval.tv_sec);
    micros_used= ((secs_used*1000000) + end_timeval.tv_usec) - (start_timeval.tv_usec);
    return (micros_used/1e6);
}



int main(int argc, char** argv){
    int i, j, m, n, k, p, q, s, vnum, frank, culaVersion;
    double normM,normU,normS,normV,normP,percent_error,elapsed_secs;
    mat *M, *U, *S, *V, *P;
    struct timeval start_timeval, end_timeval;
    char *M_file = "../../data/A_mat_6kx12k.bin";

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
    k = 5800;
    p = 100;
    q = 2;
    s = 1;
    vnum = 1;
    U = matrix_new(m,k);
    S = matrix_new(k,k);
    V = matrix_new(n,k);
    
    printf("calling random SVD with k = %d\n", k);
    gettimeofday(&start_timeval, NULL);
    //randomized_low_rank_svd1(M, k, U, S, V);
    //randomized_low_rank_svd2(M, k, U, S, V);
    low_rank_svd_rand_decomp_fixed_rank(M, k, p, vnum, q, s, &frank, &U, &S, &V);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("elapsed time: about %4.2f seconds\n", elapsed_secs);

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

