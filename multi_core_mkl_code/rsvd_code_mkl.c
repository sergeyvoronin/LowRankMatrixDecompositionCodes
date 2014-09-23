/* QR code with socket server
Sergey Voronin - 2014  */

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <strings.h>
#include "mkl.h"
#include "mkl_lapacke.h"
#include "mkl_vsl.h"


#define SEED    777
#define BRNG    VSL_BRNG_MCG31
#define METHOD  VSL_RNG_METHOD_GAUSSIAN_ICDF


typedef struct {
    int nrows, ncols;
    double * d;
} mat;



typedef struct {
    int nrows;
    double * d;
} vec;


/* initialize new matrix and set all entries to zero */
mat * matrix_new(int nrows, int ncols)
{
    mat *M = malloc(sizeof(mat));
    //M->d = (double*)mkl_malloc(sizeof(double) * nrows*ncols, 64);
    //M->d = (double*)mkl_calloc(nrows*ncols, sizeof(double), 64);
    M->d = (double*)calloc(nrows*ncols, sizeof(double));
    M->nrows = nrows;
    M->ncols = ncols;
    return M;
}


/* initialize new vector and set all entries to zero */
vec * vector_new(int nrows)
{
    vec *v = malloc(sizeof(vec));
    //v->d = (double*)mkl_calloc(nrows,sizeof(double), 64);
    v->d = (double*)calloc(nrows,sizeof(double));
    v->nrows = nrows;
    return v;
}


void matrix_delete(mat *M)
{
    //mkl_free(M->d);
    free(M->d);
    free(M);
}


void vector_delete(vec *v)
{
    //mkl_free(v->d);
    free(v->d);
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
    #pragma omp parallel shared(v) private(i) 
    {
    #pragma omp for
    for(i=0; i<(v->nrows); i++){
        v->d[i] = data[i];
    }
    }
}


/* scale vector by a constant */
void vector_scale(vec *v, double scalar){
    int i;
    #pragma omp parallel shared(v,scalar) private(i) 
    {
    #pragma omp for
    for(i=0; i<(v->nrows); i++){
        v->d[i] = scalar*(v->d[i]);
    }
    }
}


/* scale matrix by a constant */
void matrix_scale(mat *M, double scalar){
    int i;
    #pragma omp parallel shared(M,scalar) private(i) 
    {
    #pragma omp for
    for(i=0; i<((M->nrows)*(M->ncols)); i++){
        M->d[i] = scalar*(M->d[i]);
    }
    }
}


/* compute euclidean norm of vector */
double vector_get2norm(vec *v){
    int i;
    double val, normval = 0;
    //#pragma omp parallel for
    #pragma omp parallel shared(v,normval) private(i,val) 
    {
    #pragma omp for reduction(+:normval)
    for(i=0; i<(v->nrows); i++){
        val = v->d[i];
        normval += val*val;
    }
    }
    return sqrt(normval);
}



/* copy contents of vec s to d  */
void vector_copy(vec *d, vec *s){
    int i;
    //#pragma omp parallel for
    #pragma omp parallel shared(d,s) private(i) 
    {
    #pragma omp for 
    for(i=0; i<(s->nrows); i++){
        d->d[i] = s->d[i];
    }
    }
}


/* copy contents of mat S to D  */
void matrix_copy(mat *D, mat *S){
    int i;
    //#pragma omp parallel for
    #pragma omp parallel shared(D,S) private(i) 
    {
    #pragma omp for 
    for(i=0; i<((S->nrows)*(S->ncols)); i++){
        D->d[i] = S->d[i];
    }
    }
}



/* hard threshold matrix entries  */
void matrix_hard_threshold(mat *M, double TOL){
    int i;
    #pragma omp parallel shared(M) private(i) 
    {
    #pragma omp for 
    for(i=0; i<((M->nrows)*(M->ncols)); i++){
        if(fabs(M->d[i]) < TOL){
            M->d[i] = 0;
        }
    }
    }
}


/* build transpose of matrix : Mt = M^T */
void matrix_build_transpose(mat *Mt, mat *M){
    int i,j;
    for(i=0; i<(M->nrows); i++){
        for(j=0; j<(M->ncols); j++){
            matrix_set_element(Mt,j,i,matrix_get_element(M,i,j)); 
        }
    }
}



/* append matrices side by side: C = [A, B] */
void append_matrices_horizontally(mat *A, mat *B, mat *C){
    int i,j;

    for(i=0; i<A->nrows; i++){
        for(j=0; j<A->ncols; j++){
            matrix_set_element(C,i,j,matrix_get_element(A,i,j));
        }
    }

    for(i=0; i<B->nrows; i++){
        for(j=0; j<B->ncols; j++){
            matrix_set_element(C,i,A->ncols + j,matrix_get_element(B,i,j));
        }
    }

    /*
    for(i=0; i<((A->nrows)*(A->ncols)); i++){
        C->d[ind] = A->d[i];
        ind++;
    }

    for(i=0; i<((B->nrows)*(B->ncols)); i++){
        C->d[ind] = B->d[i];
        ind++;
    }*/
}



/* append matrices vertically: C = [A; B] */
void append_matrices_vertically(mat *A, mat *B, mat *C){
    int i,j;

    for(i=0; i<A->nrows; i++){
        for(j=0; j<A->ncols; j++){
            matrix_set_element(C,i,j,matrix_get_element(A,i,j));
        }
    }

    for(i=0; i<B->nrows; i++){
        for(j=0; j<B->ncols; j++){
            matrix_set_element(C,A->nrows+i,j,matrix_get_element(B,i,j));
        }
    }
}


/* copy contents of mat S to D  
TO UPDATE */
void matrix_copy_first_columns(mat *D, mat *S, int num_columns){
    int i,j;
    for(i=0; i<(S->nrows); i++){
        for(j=0; j<num_columns; j++){
            matrix_set_element(D,i,j,matrix_get_element(S,i,j));
        }
    }
}


/* subtract b from a and save result in a  */
void vector_sub(vec *a, vec *b){
    int i;
    //#pragma omp parallel for
    #pragma omp parallel shared(a,b) private(i) 
    {
    #pragma omp for 
    for(i=0; i<(a->nrows); i++){
        a->d[i] = a->d[i] - b->d[i];
    }
    }
}


/* subtract B from A and save result in A  */
void matrix_sub(mat *A, mat *B){
    int i;
    //#pragma omp parallel for
    #pragma omp parallel shared(A,B) private(i) 
    {
    #pragma omp for 
    for(i=0; i<((A->nrows)*(A->ncols)); i++){
        A->d[i] = A->d[i] - B->d[i];
    }
    }
}


/* matrix frobenius norm */
double get_matrix_frobenius_norm(mat *M){
    int i;
    double val, normval = 0;
    #pragma omp parallel shared(M,normval) private(i,val) 
    {
    #pragma omp for reduction(+:normval)
    for(i=0; i<((M->nrows)*(M->ncols)); i++){
        val = M->d[i];
        normval += val*val;
    }
    }
    return sqrt(normval);
}


/* matrix max abs val */
double get_matrix_max_abs_element(mat *M){
    int i;
    double val, max = 0;
    for(i=0; i<((M->nrows)*(M->ncols)); i++){
        val = M->d[i];
        if( abs(val) > max )
            max = val;
    }
    return max;
}



/* load matrix from text file 
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
    printf("num_nnz: %d\n", num_nonzeros);
    printf("initializing M of size %d by %d\n", num_rows, num_columns);
    M = matrix_new(num_rows,num_columns);
    printf("done..\n");

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


/* C = A*B ; column major */
void matrix_matrix_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A->nrows, B->ncols, A->ncols, alpha, A->d, A->ncols, B->d, B->ncols, beta, C->d, C->ncols);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A->nrows, B->ncols, A->ncols, alpha, A->d, A->nrows, B->d, B->nrows, beta, C->d, C->nrows);
}


/* C = A^T*B ; column major */
void matrix_transpose_matrix_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, A->ncols, B->ncols, A->nrows, alpha, A->d, A->ncols, B->d, B->ncols, beta, C->d, C->ncols);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, A->ncols, B->ncols, A->nrows, alpha, A->d, A->nrows, B->d, B->nrows, beta, C->d, C->nrows);
}


/* C = A*B^T ; column major */
void matrix_matrix_transpose_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A->nrows, B->nrows, A->ncols, alpha, A->d, A->ncols, B->d, B->ncols, beta, C->d, C->ncols);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A->nrows, B->nrows, A->ncols, alpha, A->d, A->nrows, B->d, B->nrows, beta, C->d, C->nrows);
}


/* y = M*x ; column major */
void matrix_vector_mult(mat *M, vec *x, vec *y){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_dgemv (CblasColMajor, CblasNoTrans, M->nrows, M->ncols, alpha, M->d, M->nrows, x->d, 1, beta, y->d, 1);
}


/* y = M^T*x ; column major */
void matrix_transpose_vector_mult(mat *M, vec *x, vec *y){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_dgemv (CblasColMajor, CblasTrans, M->nrows, M->ncols, alpha, M->d, M->nrows, x->d, 1, beta, y->d, 1);
}



/* set column of matrix to vector */
void matrix_set_col(mat *M, int j, vec *column_vec){
    int i;
    #pragma omp parallel shared(column_vec,M,j) private(i) 
    {
    #pragma omp for
    for(i=0; i<M->nrows; i++){
        matrix_set_element(M,i,j,vector_get_element(column_vec,i));
    }
    }
}


/* extract column of a matrix into a vector */
void matrix_get_col(mat *M, int j, vec *column_vec){
    int i;
    #pragma omp parallel shared(column_vec,M,j) private(i) 
    {
    #pragma omp parallel for
    for(i=0; i<M->nrows; i++){ 
        vector_set_element(column_vec,i,matrix_get_element(M,i,j));
    }
    }
}


/* extract row i of a matrix into a vector */
void matrix_get_row(mat *M, int i, vec *row_vec){
    int j;
    #pragma omp parallel shared(row_vec,M,i) private(j) 
    {
    #pragma omp parallel for
    for(j=0; j<M->ncols; j++){ 
        vector_set_element(row_vec,j,matrix_get_element(M,i,j));
    }
    }
}


/* put vector row_vec as row i of a matrix */
void matrix_set_row(mat *M, int i, vec *row_vec){
    int j;
    #pragma omp parallel shared(row_vec,M,i) private(j) 
    {
    #pragma omp parallel for
    for(j=0; j<M->ncols; j++){ 
        matrix_set_element(M,i,j,vector_get_element(row_vec,j));
    }
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



/* initialize a random matrix */
void initialize_random_matrix_slow(mat *M){
    int i,m,n;
    double val;
    m = M->nrows;
    n = M->ncols;

    // seed 
    srand(time(NULL));

    // read and set elements
    #pragma omp parallel shared(M,m,n) private(i,val) 
    {
    #pragma omp parallel for
    for(i=0; i<(m*n); i++){
        val = ((double) rand() / (RAND_MAX));
        M->d[i] = val;
    }
    }
}



/* initialize a random matrix */
void initialize_random_matrix(mat *M){
    int i,m,n;
    double val;
    m = M->nrows;
    n = M->ncols;
    float a=0.0,sigma=1.0;
    int N = m*n;
    float *r;
    VSLStreamStatePtr stream;
    
    r = (float*)malloc(N*sizeof(float));
   
    vslNewStream( &stream, BRNG,  time(NULL) );
    //vslNewStream( &stream, BRNG,  SEED );

    vsRngGaussian( METHOD, stream, N, r, a, sigma );

    // read and set elements
    #pragma omp parallel shared(M,N,r) private(i,val) 
    {
    #pragma omp parallel for
    for(i=0; i<N; i++){
        val = r[i];
        M->d[i] = val;
    }
    }
    
    free(r);
}



/* compute evals and evecs of symmetric matrix M
*/
void compute_evals_and_evecs_of_symm_matrix(mat *S, vec *evals){
    //LAPACKE_dsyev( LAPACK_ROW_MAJOR, 'V', 'U', S->nrows, S->d, S->nrows, evals->d);
    LAPACKE_dsyev( LAPACK_COL_MAJOR, 'V', 'U', S->nrows, S->d, S->ncols, evals->d);
}


/* initialize diagonal matrix from vector data */
void initialize_diagonal_matrix(mat *D, vec *data){
    int i;
    #pragma omp parallel shared(D) private(i)
    { 
    #pragma omp parallel for
    for(i=0; i<(D->nrows); i++){
        matrix_set_element(D,i,i,data->d[i]);
    }
    }
}



/* initialize identity */
void initialize_identity_matrix(mat *D){
    int i;
    #pragma omp parallel shared(D) private(i)
    { 
    #pragma omp parallel for
    for(i=0; i<(D->nrows); i++){
        matrix_set_element(D,i,i,1.0);
    }
    }
}



/* invert diagonal matrix */
void invert_diagonal_matrix(mat *Dinv, mat *D){
    int i;
    #pragma omp parallel shared(D,Dinv) private(i)
    {
    #pragma omp parallel for
    for(i=0; i<(D->nrows); i++){
        matrix_set_element(Dinv,i,i,1.0/(matrix_get_element(D,i,i)));
    }
    }
}



/* returns the dot product of two vectors */
double vector_dot_product(vec *u, vec *v){
    int i;
    double dotval = 0;
    #pragma omp parallel shared(u,v,dotval) private(i) 
    {
    #pragma omp for reduction(+:dotval)
    for(i=0; i<u->nrows; i++){
        dotval += (u->d[i])*(v->d[i]);
    }
    }
    return dotval;
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
            matrix_get_col(Q, j, vj);
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
    printf("doing QR with m = %d, n = %d, k = %d\n", m,n,k);
    mat *R_full = matrix_new(m,n);
    matrix_copy(R_full,M);
    //vec *tau = vector_new(n);
    vec *tau = vector_new(k);

    // get R
    //printf("get R..\n");
    //LAPACKE_dgeqrf(CblasColMajor, m, n, R_full->d, n, tau->d);
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, R_full->nrows, R_full->ncols, R_full->d, R_full->nrows, tau->d);
    
    for(i=0; i<k; i++){
        for(j=0; j<k; j++){
            if(j>=i){
                matrix_set_element(R,i,j,matrix_get_element(R_full,i,j));
            }
        }
    }

    // get Q
    matrix_copy(Q,R_full); 
    //printf("dorgqr..\n");
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q->nrows, Q->ncols, min(Q->ncols,Q->nrows), Q->d, Q->nrows, tau->d);

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
    vec *tau = vector_new(k);

    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, Q->d, m, tau->d);
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, Q->d, m, tau->d);

    // clean up
    vector_delete(tau);
}



/* computes SVD: M = U*S*V^T; note Vt is V transposed */
void singular_value_decomposition(mat *M, mat *U, mat *S, mat *Vt){
    int m,n,k;
    m = M->nrows; n = M->ncols;
    k = min(m,n);
    //vec * work = vector_new(max(m,n));
    vec * work = vector_new(k);
    vec * svals = vector_new(k);

    //LAPACKE_dgesvd( LAPACK_COL_MAJOR, 'A', 'A', m, n, M->d, n, svals->d, U->d, n, Vt->d, max(m,n), work->d );
    LAPACKE_dgesvd( LAPACK_COL_MAJOR, 'A', 'A', k, k, M->d, k, svals->d, U->d, k, Vt->d, k, work->d );
   // LAPACKE_dgesvd( LAPACK_COL_MAJOR, 'A', 'A', m, n, M->d, m, svals->d, U->d, m, Vt->d, m, work->d );

    //printf("init diagonal..\n");
    initialize_diagonal_matrix(S, svals);

    printf("norm(U) = %f\n", get_matrix_frobenius_norm(U));
    printf("norm(S) = %f\n", get_matrix_frobenius_norm(S));
    printf("norm(Vt) = %f\n", get_matrix_frobenius_norm(Vt));

    vector_delete(work);
    vector_delete(svals);
}


/* the following functions are used for computing ID later */
void invert_upper_triangular_matrix(mat *Minv){
    //LAPACKE_dtrtri( LAPACK_ROW_MAJOR, 'U', 'N', Minv->ncols, Minv->d, Minv->nrows);
    LAPACKE_dtrtri( LAPACK_COL_MAJOR, 'U', 'N', Minv->nrows, Minv->d, Minv->nrows);
}



void fill_matrix_from_first_rows(mat *M, int k, mat *M_k){
    int i;
    vec *row_vec;
    //#pragma omp parallel shared(M,M_k,k) private(i,row_vec) 
    {
    //#pragma omp for
    for(i=0; i<k; i++){
        row_vec = vector_new(M->ncols);
        matrix_get_row(M,i,row_vec);
        matrix_set_row(M_k,i,row_vec);
        vector_delete(row_vec);
    }
    }
}


void fill_matrix_from_first_columns(mat *M, int k, mat *M_k){
    int i;
    vec *col_vec;
    //#pragma omp parallel shared(M,M_k,k) private(i,col_vec) 
    {
    //#pragma omp for
    for(i=0; i<k; i++){
        col_vec = vector_new(M->nrows);
        matrix_get_col(M,i,col_vec);
        matrix_set_col(M_k,i,col_vec);
        vector_delete(col_vec);
    }
    }
}


void fill_matrix_from_last_columns(mat *M, int k, mat *M_k){
    int i,ind;
    vec *col_vec;
    ind = 0;
    for(i=k; i<M->ncols; i++){
        col_vec = vector_new(M->nrows);
        matrix_get_col(M,i,col_vec);
        matrix_set_col(M_k,ind,col_vec);
        vector_delete(col_vec);
        ind++;
    }
}


void fill_matrix_from_column_list(mat *M, vec *I, mat *M_k){
    int i,col_num;
    vec *col_vec;
    for(i=0; i<(M_k->ncols); i++){
        col_num = vector_get_element(I,i);
        col_vec = vector_new(M->nrows);
        matrix_get_col(M,col_num,col_vec);
        matrix_set_col(M_k,i,col_vec);
        vector_delete(col_vec);
    }
}



void fill_matrix_from_row_list(mat *M, vec *I, mat *M_k){
    int i,row_num;
    vec *row_vec;
    for(i=0; i<(M_k->nrows); i++){
        row_num = vector_get_element(I,i);
        row_vec = vector_new(M->ncols);
        matrix_get_row(M,row_num,row_vec);
        matrix_set_row(M_k,i,row_vec);
        vector_delete(row_vec);
    }
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
    normA = get_matrix_frobenius_norm(A);
    normB = get_matrix_frobenius_norm(B);
    normA_minus_B = get_matrix_frobenius_norm(A_minus_B);
    return 100.0*normA_minus_B/normA;
}


/* solve A X = B where A is upper triangular matrix and X is a matrix 
invert different ways
1. using tridiagonal matrix system solve
2. using inverse of tridiagonal matrix solve
3. Use SVD of A to compute inverse 
default: solve column by column with tridiagonal system
*/
void upper_triangular_system_solve(mat *A, mat *B, mat *X, int solve_type){
    int j;
    double alpha = 1.0;
    vec *col_vec;
    mat *S;

    //printf("A is %d by %d\n", A->nrows, A->ncols);
    //printf("X is %d by %d\n", X->nrows, X->ncols);
    //printf("B is %d by %d\n", B->nrows, B->ncols);

    if(solve_type == 1){
        S = matrix_new(B->nrows,B->ncols);
        matrix_copy(S,B);
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, B->nrows, B->ncols, alpha, A->d, A->nrows, S->d, S->nrows);
        matrix_copy(X,S);
        matrix_delete(S);
    }
    else if(solve_type == 2){
        invert_upper_triangular_matrix(A);
        matrix_matrix_mult(A,B,X);
    }
    else if(solve_type == 3){
        mat *U, *S, *Sinv, *Vt, *SinvUt, *VSinvUt;
        U = matrix_new(A->nrows, A->nrows);
        S = matrix_new(A->nrows, A->nrows);
        Sinv = matrix_new(A->nrows, A->nrows);
        Vt = matrix_new(A->nrows, A->nrows);
        SinvUt = matrix_new(A->nrows, A->nrows);
        VSinvUt = matrix_new(A->nrows, A->nrows);
        singular_value_decomposition(A, U, S, Vt);
        invert_diagonal_matrix(Sinv,S);
        matrix_matrix_transpose_mult(Sinv,U,SinvUt); 
        matrix_transpose_matrix_mult(Vt,SinvUt,VSinvUt);
        matrix_matrix_mult(VSinvUt, B, X);
    }
    else{
        col_vec = vector_new(B->nrows);
        for(j=0; j<B->ncols; j++){
            matrix_get_col(B,j,col_vec);
            cblas_dtrsv (CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, A->ncols, A->d, A->ncols, col_vec->d, 1);
            matrix_set_col(X,j,col_vec);     
        }
        vector_delete(col_vec);
    }
}


double get_matrix_column_norm_squared(mat *M, int colnum){
    int i, m, n;
    double val,colnorm;
    m = M->nrows;
    n = M->ncols;
    colnorm = 0;
    for(i=0; i<m; i++){
        val = matrix_get_element(M,i,colnum);
        colnorm += val*val;
    }
    return colnorm;
}


double matrix_getmaxcolnorm(mat *M){
    int i,m,n;
    vec *col_vec;
    double vecnorm, maxnorm;
    m = M->nrows; n = M->ncols;
    col_vec = vector_new(m);

    maxnorm = 0;    
    #pragma omp parallel for
    for(i=0; i<n; i++){
        matrix_get_col(M,i,col_vec);
        vecnorm = vector_get2norm(col_vec);
        #pragma omp critical
        if(vecnorm > maxnorm){
            maxnorm = vecnorm;
        }
    }

    vector_delete(col_vec);
    return maxnorm;
}


void compute_matrix_column_norms(mat *M, vec *column_norms){
    int i,m,n;
    m = M->nrows; n = M->ncols;
    for(i=0; i<n; i++){
        vector_set_element(column_norms,i, get_matrix_column_norm_squared(M,i)); 
    }
}




void get_householder_matrix(vec *x, int ind1, int ind2, mat *H){
    int i,j;
    double val,normval;

    //printf("---------start get_house---------\n");


    for(i=ind1; i<ind2; i++){
        matrix_set_element(H,i,0,vector_get_element(x,i));
    }

    normval = 0;
    for(i=ind1; i<ind2; i++){
        val = vector_get_element(x,i);
        normval += val*val;
    }
    normval = sqrt(normval);
    //printf("normval = %f\n", normval);

    matrix_set_element(H,ind1,0,matrix_get_element(H,ind1,0) - normval);

    //printf("after subtract with ind1 = %d:\n", ind1);
    //matrix_print(H);

    val = get_matrix_frobenius_norm(H);
    if(val > 0){
        matrix_scale(H, sqrt(2/(val*val)));
    }

    //printf("---------end get_house---------\n");
}



/* computes the approximate pivoted QR factorization of speicifed rank 
: A \approx Qk*Rk*P^T */
void pivoted_QR_of_specified_rank(mat *M, int k, int *frank, mat **Qk, mat **Rk, mat **P, vec **I){
    int i,j,ind,tmpind,m,n,max_colnorm_index,column_norm_zero,loop_lim,checkQRresult;
    double max_colnorm,tempval,val1,val2,val3;
    int * max_colnorm_indices; 
    mat *Q, *R, *Rk1, *Rk2, *invRk1, *H, *HHt, *HtR, *HHtR, *QHHt, *T, *Ik;
    vec *column_norms1, *column_norms2, *row_vec1, *row_vec2, *col_vec1, *col_vec2, *Ientries;
    m = M->nrows;
    n = M->ncols;

    Q = matrix_new(m,m);
    R = matrix_new(m,n);
    *P = matrix_new(n,n);
    H = matrix_new(m,1);
    HHt = matrix_new(m,m);
    QHHt = matrix_new(m,m);
    HtR = matrix_new(1,n);
    HHtR = matrix_new(m,n);

    matrix_copy(R,M);
    
    initialize_identity_matrix(Q);
    initialize_identity_matrix(*P);
    
    column_norms1 = vector_new(n);
    column_norms2 = vector_new(n);
    compute_matrix_column_norms(M,column_norms1); 

    // loop parameters
    column_norm_zero = 0;
    *frank = 0;

    for(i=0; !column_norm_zero && i<k; i++){
        
        //if( i%50 == 0 ){
        printf("=========> iteration %d of %d--->\n", i+1, k);
        //}
        
        //printf("find max_colnorm..\n");
        max_colnorm = vector_get_element(column_norms1,i);
        max_colnorm_index = i;
        for(j=(i+1); j<n; j++){
            if(vector_get_element(column_norms1,j) > max_colnorm){
                max_colnorm_index = j;
                max_colnorm = vector_get_element(column_norms1,j);   
            }
        }
        printf("max_colnorm = %f\n", max_colnorm);
        printf("max_colnorm_index(+1) = %d\n", max_colnorm_index + 1);


        /* break; Qk and Rk already set for i>0 */
        if(vector_get_element(column_norms1,max_colnorm_index) == 0 && i>0){
            column_norm_zero = 1;
            printf("column norm zero detected and will break!\n");
            break;
        }
        else{
            *frank = i+1;
        }

        // swap P columns
        //printf("swap P..\n");
        col_vec1 = vector_new((*P)->ncols);
        col_vec2 = vector_new((*P)->ncols);
        matrix_get_col(*P,i,col_vec1);
        matrix_get_col(*P,max_colnorm_index,col_vec2);
        matrix_set_col(*P,i,col_vec2);
        matrix_set_col(*P,max_colnorm_index,col_vec1); 
        vector_delete(col_vec1); 
        vector_delete(col_vec2); 

        // swap R columns
        //printf("swap R..\n");
        col_vec1 = vector_new(R->nrows);
        col_vec2 = vector_new(R->nrows);
        matrix_get_col(R,i,col_vec1);
        matrix_get_col(R,max_colnorm_index,col_vec2);
        matrix_set_col(R,i,col_vec2);
        matrix_set_col(R,max_colnorm_index,col_vec1); 
        vector_delete(col_vec1); 
        vector_delete(col_vec2); 

        
        // apply Pt to column norms
        //printf("apply Pt..\n");
        matrix_transpose_vector_mult(*P,column_norms1,column_norms2);
        vector_copy(column_norms1,column_norms2);
        //printf("after swap column_norms1\n");
        //vector_print(column_norms1);


        // get H and transform R and Q
        printf("get house iteration %d..\n", i);
        col_vec1 = vector_new(R->nrows);
        matrix_get_col(R,i,col_vec1);
        matrix_scale(H,0);
        get_householder_matrix(col_vec1, i, m, H);
        vector_delete(col_vec1); 
        printf("norm(H,fro) = %f\n", get_matrix_frobenius_norm(H));

        //printf("apply H to R..\n");
        printf("apply transformations to R and Q..\n");
        matrix_transpose_matrix_mult(H,R,HtR);
        matrix_matrix_mult(H,HtR,HHtR);
        matrix_sub(R, HHtR);
        printf("norm(R,fro) = %f\n", get_matrix_frobenius_norm(R));

        //printf("apply H to Q..\n");
        matrix_matrix_transpose_mult(H,H,HHt);
        matrix_matrix_mult(Q,HHt,QHHt);
        matrix_sub(Q,QHHt);
        printf("norm(Q,fro) = %f\n", get_matrix_frobenius_norm(Q));

        //printf("downdate norms..\n");
        if(i != (n-1)){
            for(ind = i+1; ind < n ; ind++){
                val1 = vector_get_element(column_norms1,ind);
                val2 = matrix_get_element(R,i,ind); 
                vector_set_element(column_norms1,ind,val1 - val2*val2);
            } 
        }
        //printf("column_norms1:\n");
        //vector_print(column_norms1);
    }        

    //printf("construct Qk and Rk..\n");
    *Qk = matrix_new(m,*frank);
    *Rk = matrix_new(*frank,n);

    fill_matrix_from_first_columns(Q, *frank, *Qk);
    fill_matrix_from_first_rows(R, *frank, *Rk);

    //construct vector I
    *I = vector_new(n);
    Ientries = vector_new(n);
    for(i=0; i<n; i++){
        vector_set_element(Ientries,i,i+1);
    }
    matrix_transpose_vector_mult(*P,Ientries,*I);

    matrix_delete(Q);
    matrix_delete(R);
    vector_delete(column_norms1);
    vector_delete(column_norms2);
}




void pivotedQR_fat_matrix_mkl(mat *M, mat **Q, mat **R, mat **P, vec **I){
    int i,j,k,m,n;
    mat *Mwork, *Porig;
    vec *col_vec;
    m = M->nrows; n = M->ncols;
    k = min(m,n);
    
    Mwork = matrix_new(m,n);
    matrix_copy(Mwork,M);

    int *Iarr = (int*)malloc(n*sizeof(int));
    double *tau_arr = (double*)malloc(min(m,n)*sizeof(double));
    
    // get R
    LAPACKE_dgeqp3(CblasColMajor, Mwork->nrows, Mwork->ncols, Mwork->d, Mwork->nrows, Iarr, tau_arr);

    *R = matrix_new(m,n);
    for(i=0; i<m; i++){
        for(j=i; j<n; j++){
            matrix_set_element(*R,i,j,matrix_get_element(Mwork,i,j));
        }
    }

    // get Q
    LAPACKE_dorgqr(CblasColMajor, Mwork->nrows, Mwork->nrows, min(Mwork->nrows,Mwork->ncols), Mwork->d, Mwork->nrows, tau_arr);

    *Q = matrix_new(k,k);
    for(i=0; i<k; i++){
        for(j=0; j<k; j++){
            matrix_set_element(*Q,i,j,matrix_get_element(Mwork,i,j));
        }
    }

    // get I
    *I = vector_new(n);
    for(i=0; i<n; i++){
        vector_set_element(*I,i,Iarr[i]-1);
    }

    // get P
    Porig = matrix_new(n,n);
    *P = matrix_new(n,n);
    initialize_identity_matrix(Porig);
    col_vec = vector_new(n);
    for(i=0; i<n; i++){
        matrix_get_col(Porig,vector_get_element(*I,i),col_vec);
        matrix_set_col(*P,i,col_vec);
    }

    matrix_delete(Porig);
    vector_delete(col_vec);
}



/* computes the approximate ID decomposition of a matrix of specified rank 
: [I,T] = id_decomp(M,k) 
where I is the vector from the permutation and T = inv(Rk1)*Rk2 */
void id_decomposition_of_specified_rank(mat *M, int k, int *frank, vec **I, mat **T, mat **P){
    int i,j,frankQR,ind,tmpind,m,n,max_colnorm_index,column_norm_zero,loop_lim,checkQRresult;
    mat *Qk, *Rk, *Rk1, *Rk2, *invRk1;
    vec *Imultvec;
    m = M->nrows;
    n = M->ncols;

    pivoted_QR_of_specified_rank(M, k, &frankQR, &Qk, &Rk, P, I);
    *frank = frankQR;
    printf("norm(Qk,fro) = %f\n", get_matrix_frobenius_norm(Qk));
    printf("norm(Rk,fro) = %f\n", get_matrix_frobenius_norm(Rk));
    printf("norm(P,fro) = %f\n", get_matrix_frobenius_norm(*P));

    printf("construct Rk1 and Rk2..\n");
    Rk1 = matrix_new(frankQR,frankQR);
    Rk2 = matrix_new(frankQR,n-frankQR);
    fill_matrix_from_first_columns(Rk, frankQR, Rk1);
    fill_matrix_from_last_columns(Rk, frankQR, Rk2);

    printf("sizes of Rk1 and Rk2:\n");
    printf("Rk1: %d by %d\n", Rk1->nrows, Rk1->ncols);
    printf("Rk2: %d by %d\n", Rk2->nrows, Rk2->ncols);
    printf("norm(Rk1,fro) = %f\n", get_matrix_frobenius_norm(Rk1));
    printf("norm(Rk2,fro) = %f\n", get_matrix_frobenius_norm(Rk2));

    invRk1 = matrix_new(Rk1->nrows,Rk1->ncols);
    *T = matrix_new(Rk2->nrows,Rk2->ncols);

    // NOTE: must enforce Rk1 to be symmetric before inverting..
    // %Rk1*T = Rk2
    matrix_copy_symmetric(invRk1,Rk1);
    printf("norm(invRk1,fro) = %f\n", get_matrix_frobenius_norm(invRk1));
    upper_triangular_system_solve(invRk1,Rk2,*T,1);
    printf("norm(*T,fro) = %f\n", get_matrix_frobenius_norm(*T));

    //matrix_hard_threshold(invRk1,1e-4);
    //printf("construct invRk1\n");
    //invert_upper_triangular_matrix(invRk1);
    //printf("norm(invRk1,fro) = %f\n", get_matrix_frobenius_norm(invRk1));

    //printf("T = invRk1 * Rk2\n");
    //matrix_matrix_mult(invRk1,Rk2,*T);
    

    //I = P'*[1:size(P,1)]';
    /*printf("construct I..\n");
    *I = vector_new(M->ncols);
    Imultvec = vector_new(M->ncols);
    for(i=0; i<(*I)->nrows; i++){
        vector_set_element(Imultvec,i,i);
    }
    matrix_transpose_vector_mult(*P,Imultvec,*I);*/

    vector_delete(Imultvec);
    printf("end function\n");
}



/* computes the approximate ID decomposition of a matrix of specified rank 
: [I,T] = id_decomp(M,k) 
where I is the vector from the permutation and T = inv(Rk1)*Rk2 */
void id_decomposition(mat *M, int solvetype, vec **I, mat **T, mat **P){
    int i,j,frankQR,ind,tmpind,m,n,max_colnorm_index,column_norm_zero,loop_lim,checkQRresult;
    mat *Qk, *Rk, *Rk1, *Rk2, *Rk1upt;
    vec *Imultvec;
    m = M->nrows;
    n = M->ncols;
    frankQR = min(m,n);
    
    pivotedQR_fat_matrix_mkl(M, &Qk, &Rk, P, I);
    printf("norm(Qk,fro) = %f\n", get_matrix_frobenius_norm(Qk));
    printf("norm(Rk,fro) = %f\n", get_matrix_frobenius_norm(Rk));
    printf("norm(P,fro) = %f\n", get_matrix_frobenius_norm(*P));

    printf("construct Rk1 and Rk2..\n");
    Rk1 = matrix_new(frankQR,frankQR);
    Rk2 = matrix_new(frankQR,n-frankQR);
    fill_matrix_from_first_columns(Rk, frankQR, Rk1);
    fill_matrix_from_last_columns(Rk, frankQR, Rk2);

    printf("sizes of Rk1 and Rk2:\n");
    printf("Rk1: %d by %d\n", Rk1->nrows, Rk1->ncols);
    printf("Rk2: %d by %d\n", Rk2->nrows, Rk2->ncols);
    printf("norm(Rk1,fro) = %f\n", get_matrix_frobenius_norm(Rk1));
    printf("norm(Rk2,fro) = %f\n", get_matrix_frobenius_norm(Rk2));

    *T = matrix_new(Rk2->nrows,Rk2->ncols);

    // NOTE: must enforce Rk1 to be symmetric before inverting..
    // %Rk1*T = Rk2
    Rk1upt = matrix_new(Rk1->nrows,Rk1->ncols);
    matrix_copy_symmetric(Rk1upt,Rk1);
    printf("norm(Rk1upt,fro) = %f\n", get_matrix_frobenius_norm(Rk1upt));
    upper_triangular_system_solve(Rk1upt,Rk2,*T,solvetype);

    printf("end function\n");
}




/* computes approximation to matrix through the randomized ID procedure */
void randomized_id_approximation1(mat *M, int k, int solvetype){
    int i,j,m,n,myt,nyt,kyt,frank;
    double val;
    vec *I;
    mat *RN, *Y, *Yt, *T, *P, *Tt, *Ik, *U1, *U, *MI, *MA;
    m = M->nrows; n = M->ncols;

    // build random matrix
    RN = matrix_new(n, k);
    printf("form RN..\n");
    initialize_random_matrix(RN);
    //RN = matrix_load_from_binary_file("data/Omega.bin");
    //printf("norm(RN,fro) = %f\n", get_matrix_frobenius_norm(RN));

    // multiply to get matrix of random samples Y
    printf("form Y..\n");
    Y = matrix_new(m,k);
    matrix_matrix_mult(M, RN, Y);

    printf("form Yt..\n");
    Yt = matrix_new(k,m);
    matrix_build_transpose(Yt,Y);
    //printf("norm(Yt,fro) = %f\n", get_matrix_frobenius_norm(Yt));

    myt = Yt->nrows;
    nyt = Yt->ncols;
    frank = min(myt,nyt);

    printf("do ID decomp..\n");
    id_decomposition(Yt, solvetype, &I, &T, &P);
    printf("norm(T,fro) = %f\n", get_matrix_frobenius_norm(T));
    printf("norm(T,max) = %f\n", get_matrix_max_abs_element(T));
    
    Tt = matrix_new(T->ncols,T->nrows);
    matrix_build_transpose(Tt,T);
    //printf("norm(Tt,fro) = %f\n", get_matrix_frobenius_norm(Tt));

    Ik = matrix_new(frank,frank);
    initialize_identity_matrix(Ik);
    //printf("norm(Ik,fro) = %f\n", get_matrix_frobenius_norm(Ik));

    printf("append mats vertically..\n");
    U1 = matrix_new(Ik->nrows + Tt->nrows,Ik->ncols);
    append_matrices_vertically(Ik,Tt,U1);
    //printf("norm(U1,fro) = %f\n", get_matrix_frobenius_norm(U1));

    printf("U = P*U1..\n");
    printf("P is %d by %d\n", P->nrows, P->ncols);
    printf("U1 is %d by %d\n", U1->nrows, U1->ncols);
    U = matrix_new(P->nrows,U1->ncols);
    matrix_matrix_mult(P,U1,U);
    //printf("norm(U,fro) = %f\n", get_matrix_frobenius_norm(U));

    // MI = M(I(1:frank),:); 
    printf("fill matrix from row list..\n");
    MI = matrix_new(frank,M->ncols);
    printf("MI is %d by %d\n", MI->nrows, MI->ncols);
    fill_matrix_from_row_list(M, I, MI);
    printf("norm(MI,fro) = %f\n", get_matrix_frobenius_norm(MI));

    MA = matrix_new(m,n);
    matrix_matrix_mult(U,MI,MA);

    printf("U is %d by %d\n", U->nrows, U->ncols);
    printf("MI is %d by %d\n", MI->nrows, MI->ncols);

    printf("norm(M,fro) = %f\n", get_matrix_frobenius_norm(M));
    printf("norm(MA,fro) = %f\n", get_matrix_frobenius_norm(MA));
    printf("percent error = %f\n", get_percent_error_between_two_mats(M,MA));
}






/* computes the approximate low rank SVD of rank k of matrix M using BBt version */
void randomized_low_rank_svd1(mat *M, int k, mat **U, mat **S, mat **V){
    int i,j,m,n;
    double val;
    m = M->nrows; n = M->ncols;

    // setup mats
    *U = matrix_new(m,k);
    *S = matrix_new(k,k);
    *V = matrix_new(n,k);

    // build random matrix
    mat *RN = matrix_new(n, k);
    printf("form RN..\n");
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
    initialize_diagonal_matrix(*S, singvals);
    
    // compute U = Q*Uhat mxk * kxk = mxk  
    printf("form U..\n");
    matrix_matrix_mult(Q,Uhat,*U);

    // compute nxk V 
    // V = B^T Uhat * Sigma^{-1}
    printf("form V..\n");
    mat *Sinv = matrix_new(k,k);
    mat *UhatSinv = matrix_new(k,k);
    invert_diagonal_matrix(Sinv,*S);
    matrix_matrix_mult(Uhat,Sinv,UhatSinv);
    matrix_matrix_mult(Bt,UhatSinv,*V);

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
void randomized_low_rank_svd2(mat *M, int k, mat **U, mat **S, mat **V){
    int i,j,m,n;
    double val;
    m = M->nrows; n = M->ncols;

    // setup mats
    *U = matrix_new(m,k);
    *S = matrix_new(k,k);
    *V = matrix_new(n,k);

    // build random matrix
    printf("form RN..\n");
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
    singular_value_decomposition(Rhat, Uhat, *S, Vhat_trans);

    // U = Q*Vhat_trans
    printf("form U..\n");
    matrix_matrix_transpose_mult(Q,Vhat_trans,*U);

    // V = Qhat*Uhat
    printf("form V..\n");
    matrix_matrix_mult(Qhat,Uhat,*V);

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



void estimate_rank_and_buildQ(mat *M, double TOL, mat **Q, int *estimated_rank){
    int m,n,i,j,ind;
    double vec_norm;
    mat *RN,*Y,*Qbig,*Qsmall;
    vec *vi,*vj,*p,*p1;
    m = M->nrows;
    n = M->ncols;
    Y = matrix_new(n,min(m,n));
    Y = matrix_new(n,min(m,n));
    Qbig = matrix_new(m,min(m,n));
    vi = vector_new(m);
    vj = vector_new(m);
    p = vector_new(m);
    p1 = vector_new(m);

    // build random matrix
    printf("form RN..\n");
    RN = matrix_new(n, min(m,n));
    initialize_random_matrix(RN);

    // multiply to get matrix of random samples Y
    printf("form Y..\n");
    Y = matrix_new(m, min(m,n));
    matrix_matrix_mult(M, RN, Y);

    // estimate rank k and build Q from Y
    printf("form Qbig..\n");
    Qbig = matrix_new(m, min(m,n));

    matrix_copy(Qbig, Y);

    printf("estimate rank with TOL = %f..\n", TOL);
    *estimated_rank = min(m,n);
    //for(ind=0; ind<num_ortos; ind++){
    int forbreak = 0;
    for(j=0; !forbreak && j<n; j++){
        matrix_get_col(Qbig, j, vj);
        for(i=0; i<j; i++){
            matrix_get_col(Qbig, i, vi);
            project_vector(vj, vi, p);
            vector_sub(vj, p);
            if(vector_get2norm(p) < TOL && vector_get2norm(p1) < TOL){
                *estimated_rank = j;
                forbreak = 1;
                break;
            }
            vector_copy(p1,p);
        }
        vec_norm = vector_get2norm(vj);
        vector_scale(vj, 1.0/vec_norm);
        matrix_set_col(Qbig, j, vj);
    }
    //}

    printf("estimated rank = %d\n", *estimated_rank);

    Qsmall = matrix_new(m, *estimated_rank); 
    *Q = matrix_new(m, *estimated_rank); 
    matrix_copy_first_columns(Qsmall, Qbig, *estimated_rank);
    QR_factorization_getQ(Qsmall, *Q);
}



/* computes the approximate low rank SVD of rank k of matrix M using QR version 
automatically estimates the rank needed */
void randomized_low_rank_svd2_autorank1(mat *M, double TOL, mat **U, mat **S, mat **V){
    int i,j,m,n,k,kinit;
    double val;
    mat *Q;
    m = M->nrows; n = M->ncols;
    kinit = min(m,n);

    // estimate rank k and build Q from Y
    printf("estimating rank and building Q..\n");
    //build_orthonormal_basis_from_mat(Y,Q);
    //QR_factorization_getQ(Y, Q);
    estimate_rank_and_buildQ(M,TOL,&Q,&k);
    printf("estimated rank = %d\n", k);
    printf("norm(Q,fro) = %f\n", get_matrix_frobenius_norm(Q));

    // setup U, S, and V 
    *U = matrix_new(m,k);
    *S = matrix_new(k,k);
    *V = matrix_new(n,k);

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
    singular_value_decomposition(Rhat, Uhat, *S, Vhat_trans);

    // U = Q*Vhat_trans
    printf("form U..\n");
    matrix_matrix_transpose_mult(Q,Vhat_trans,*U);

    // V = Qhat*Uhat
    printf("form V..\n");
    matrix_matrix_mult(Qhat,Uhat,*V);

    // free stuff
    matrix_delete(Q);
    matrix_delete(Rhat);
    matrix_delete(Qhat);
    matrix_delete(Uhat);
    matrix_delete(Vhat_trans);
    matrix_delete(Bt);
}






/* computes the approximate low rank SVD of rank k of matrix M using QR version 
 * with range sampling via (M M^T)^q M R*/
void randomized_low_rank_svd3(mat *M, int k, int q, mat **U, mat **S, mat **V){
    int i,j,m,n;
    double val;
    m = M->nrows; n = M->ncols;

    // setup mats
    *U = matrix_new(m,k);
    *S = matrix_new(k,k);
    *V = matrix_new(n,k);

    // build random matrix
    mat *RN = matrix_new(n, k);
    initialize_random_matrix(RN);

    // multiply to get matrix of random samples Y
    printf("form Y..\n");
    mat *Y = matrix_new(m,k);
    matrix_matrix_mult(M, RN, Y);

    // build Q from Y
    printf("form Q with q=%d..\n",q);
    mat *Q = matrix_new(m,k);
    //build_orthonormal_basis_from_mat(Y,Q);
    QR_factorization_getQ(Y, Q);

    // now refine Q
    mat *Z = matrix_new(m,k);
    Y = matrix_new(n,k);
    mat *W = matrix_new(n,k);
    for(j=0; j<q; j++){
        printf("in loop for j=%d of %d\n", j, q);
        printf("M is %d x %d and Q is %d x %d and Y is %d x %d\n", M->nrows, M->ncols, Q->nrows, Q->ncols, Y->nrows, Y->ncols);
        printf("Y = M^T*Q..\n");
        matrix_transpose_matrix_mult(M, Q, Y);
        printf("Y is %d x %d\n", Y->nrows, Y->ncols);
        if( j%2 == 0 ){
            printf("orthogonalize Y..\n");
            QR_factorization_getQ(Y, W);
            printf("Z = M*W..\n");
            matrix_matrix_mult(M,W,Z);
        }
        else{
            printf("Z = M*Y..\n");
            matrix_matrix_mult(M,Y,Z);
        }
        if( j%2 == 0 ){
            printf("orthogonalize Z..\n");
            QR_factorization_getQ(Z, Q);
        }
    }

    // orthogonalize on exit from loop
    QR_factorization_getQ(Z, Q);

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
    singular_value_decomposition(Rhat, Uhat, *S, Vhat_trans);

    // U = Q*Vhat_trans
    printf("form U..\n");
    matrix_matrix_transpose_mult(Q,Vhat_trans,*U);

    // V = Qhat*Uhat
    printf("form V..\n");
    matrix_matrix_mult(Qhat,Uhat,*V);

    // free stuff
    matrix_delete(RN);
    matrix_delete(Y);
    matrix_delete(Q);
    matrix_delete(Z);
    matrix_delete(Rhat);
    matrix_delete(Qhat);
    matrix_delete(Uhat);
    matrix_delete(Vhat_trans);
    matrix_delete(Bt);
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


void read_elements_from_socket(int sockfd, int nelements, double *elements_all){
    int i, ind, nlen, nchunks, nleftover, chunk_num;
    double * elements_local;

    // read all elements
    if(nelements <= 10000){
        nlen = read(sockfd,elements_all,nelements*sizeof(double));
        if(nlen < 0) 
            error("ERROR writing to socket");
    }
    // read in batches of 10000
    else{
        nchunks = nelements/10000;  
        nleftover = nelements % 10000;
        printf("server: nchunks = %d and nleftover = %d\n", nchunks, nleftover);
        ind = 0; // ind into global array

        for(chunk_num=0; chunk_num < nchunks; chunk_num++){
                //printf("read chunk %d\n", chunk_num);
                elements_local = (double*)malloc(10000*sizeof(double));
                nlen = read(sockfd,elements_local,10000*sizeof(double));
                for(i=0; i<10000; i++){
                    elements_all[ind++] = elements_local[i];
                }
                free(elements_local);
        }
        //printf("read leftovers..\n");
        elements_local = (double*)malloc(nleftover*sizeof(double));
        nlen = read(sockfd,elements_local,nleftover*sizeof(double));
        for(i=0; i<nleftover; i++){
            elements_all[ind++] = elements_local[i];
        }
        free(elements_local);
    }
}



void write_elements_to_socket(int sockfd, int nelements, double *elements_all){
    int i, ind, nlen, nchunks, nleftover, chunk_num;
    double * elements_local;

    // write all elements
    if(nelements <= 10000){
        printf("transfer all at once..\n");
        nlen = write(sockfd,elements_all,nelements*sizeof(double));
        if(nlen < 0) 
            error("ERROR writing to socket");
    }
    // write in batches of 10000
    else{
        nchunks = nelements/10000;  
        nleftover = nelements % 10000;
        printf("server: nchunks = %d and nleftover = %d\n", nchunks, nleftover);

        ind = 0; // ind into global array
        for(chunk_num=0; chunk_num < nchunks; chunk_num++){
                //printf("transfer chunk %d\n", chunk_num);
                elements_local = (double*)malloc(10000*sizeof(double));
                for(i=0; i<10000; i++){
                    elements_local[i] = elements_all[ind++];
                }
                nlen = write(sockfd,elements_local,10000*sizeof(double));
                free(elements_local);
                
        }
        //printf("transfer leftovers...\n");
        elements_local = (double*)malloc(nleftover*sizeof(double));
        for(i=0; i<nleftover; i++){
            elements_local[i] = elements_all[ind++];
        }
        nlen = write(sockfd,elements_local,nleftover*sizeof(double));
        free(elements_local);
    }
}



int read_int_from_socket(int sockfd, int *element){
    int nlen, ntrials, maxtrials = 3;
    nlen = read(sockfd,element,sizeof(int));
    ntrials = 0;
    while(nlen<=0 && ntrials <= maxtrials){
        sleep(1); nlen = read(sockfd,element,sizeof(int));
        ntrials++;
    }
    return nlen;
}


int write_int_from_socket(int sockfd, int *element){
    int nlen, ntrials, maxtrials = 3;
    nlen = write(sockfd,element,sizeof(int));
    ntrials = 0;
    while(nlen<=0 && ntrials <= maxtrials){
        sleep(1); nlen = write(sockfd,element,sizeof(int));
        ntrials++;
    }
    return nlen;
}


void sleep_funct(struct timespec *tspec){
    //nanosleep(tspec,NULL);
    sleep(1);
}


int main(int argc, char *argv[])
{
    int i, j, m, n, k, frank, T_solver_num, sockfd, newsockfd, portno, clilen, nelements, stop = 0;
    int nlen, number, nchunks, nleftover, ind, chunk_num, nrows, ncols;
    double normM,normU,normS,normV,normP,percent_error;
    mat *M, *U, *S, *V, *Qk, *Rk, *P, *QkRk, *QkRkPt;
    vec *I;
    time_t start_time, end_time;
    struct sockaddr_in serv_addr, cli_addr;
    double *elements_all, *elements_local;
    FILE *fp;

    int sndbuf, rcvbuf;
    struct timespec tspec;
    
    if (argc < 2) {
         fprintf(stderr,"ERROR, no port provided\n");
         return(1);
    }

    // set up timespec for half second waits
    tspec.tv_sec  = 0;
    tspec.tv_nsec = 15000000000L;


    printf("setting up socket server..\n");
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0){ 
        error("ERROR opening socket");
        close(sockfd);
        return -1;
    }
    else{
        printf("socket initialized\n");
    }
    bzero((char *) &serv_addr, sizeof(serv_addr));
    portno = atoi(argv[1]);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    //serv_addr.sin_family = AF_LOCAL;
    //serv_addr.sin_addr.s_addr = PF_LOCAL;
    serv_addr.sin_port = htons(portno);
    if(bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0){ 
        error("ERROR on binding");
        close(sockfd);
        return -1;
    }
    listen(sockfd,5);
    clilen = sizeof(cli_addr);
    printf("binding complete\n");


    printf("looping socket server.\n");
    while(!stop){
        newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
        printf("after accept\n");
        if (newsockfd < 0) 
          error("ERROR on accept");

        // set options
        sndbuf = 5000;  /* Send buffer size */
        rcvbuf = 5000;
        //nlen = setsockopt(newsockfd,SOL_SOCKET,SO_SNDBUF, &sndbuf, sizeof(sndbuf));
        //nlen = setsockopt(newsockfd,SOL_SOCKET,SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));


        read_int_from_socket(newsockfd, &m);
        read_int_from_socket(newsockfd, &n);
        read_int_from_socket(newsockfd, &k);
        read_int_from_socket(newsockfd, &T_solver_num);

        nelements = m*n;
        printf("read nrows = %d and ncols = %d\n", m, n);
        printf("read k = %d and T_solver_num = %d\n", k, T_solver_num);

        // init space for all elements 
        elements_all = (double*)malloc(nelements*sizeof(double));

        // read all elements
        read_elements_from_socket(newsockfd, nelements, elements_all);

        // set up matrix using the read data
        M = matrix_new(m,n);
        M->d = elements_all;

        printf("setup matrix M of size %d by %d\n", m,n);
        //printf("M:\n");
        //matrix_print(M);

        //printf("calling randomized_id_approximation1..\n");
        //randomized_id_approximation1(M, k, T_solver_num);
        //printf("done with randomized_id_approximation1.\n");

        printf("calling QR decomposition with rank parameter = %d\n", k);
        mat *Qk, *Rk, *P;
        pivoted_QR_of_specified_rank(M, k, &frank, &Qk, &Rk, &P, &I);
        // threshold Rk to make it have zero elements below diagonal
        matrix_hard_threshold(Rk, 1e-12);
        printf("QR finished with frank = %d\n", frank);

        // write Qk 
        nrows = Qk->nrows; ncols = Qk->ncols;
        nelements = nrows*ncols;
        write_int_from_socket(newsockfd, &nrows);
        sleep_funct(&tspec);
        write_int_from_socket(newsockfd, &ncols);
        sleep_funct(&tspec);
        write_elements_to_socket(newsockfd, nelements, Qk->d);
        sleep_funct(&tspec);

        // write Rk
        nrows = Rk->nrows; ncols = Rk->ncols;
        nelements = nrows*ncols;
        write_int_from_socket(newsockfd, &nrows);
        sleep_funct(&tspec);
        write_int_from_socket(newsockfd, &ncols);
        sleep_funct(&tspec);
        write_elements_to_socket(newsockfd, nelements, Rk->d);
        sleep_funct(&tspec);

        // write P
        nrows = P->nrows; ncols = P->ncols;
        nelements = nrows*ncols;
        write_int_from_socket(newsockfd, &nrows);
        sleep_funct(&tspec);
        write_int_from_socket(newsockfd, &ncols);
        sleep_funct(&tspec);
        write_elements_to_socket(newsockfd, nelements, P->d);
        sleep_funct(&tspec); 

        // write I
        nrows = I->nrows;
        nelements = nrows;
        printf("server: I->nrows = %d\n", I->nrows);
        write_int_from_socket(newsockfd, &nrows);
        sleep_funct(&tspec); 
        write_elements_to_socket(newsockfd, nelements, I->d);
        sleep_funct(&tspec); 
    
        // close
        if (close(newsockfd) != 0)
            printf("server newsockfd closing failed!\n");
        else
            printf("server newsockfd successfully closed!\n");

        stop = 1;
    // end of socket loop
    }
   

    printf("done with socket server.");

    if (close(sockfd) != 0)
        printf("server sockfd closing failed!\n");
    else
        printf("server sockfd successfully closed!\n");

    matrix_delete(M);
    
    return 0;
}

