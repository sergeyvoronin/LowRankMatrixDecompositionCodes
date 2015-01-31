#include "matrix_vector_functions_gsl.h"


/* write matrix to file 
format:
% comment
num_rows num_columns num_nonzeros
row col nnz
.....
row col nnz
*/
void matrix_write_to_file(gsl_matrix *M, char *fname){
    int i, j;
    FILE *fp;
    
    fp = fopen(fname,"w");
    fprintf(fp,"MatrixMarket matrix\n"); //write comment
    fprintf(fp, "%d %d %d\n", M->size1, M->size2, (M->size1)*(M->size2));
    for(i=0; i<M->size1; i++){
        for(j=0; j<M->size2; j++){
            fprintf(fp, "%d %d %f\n", i, j, gsl_matrix_get(M,i,j));
        }
    }
    fclose(fp);
}


/* load matrix from file 
format:
% comment
num_rows num_columns num_nonzeros
row col nnz
.....
row col nnz
*/
gsl_matrix * matrix_load_from_text_file(char *fname){
    int i, j, num_rows, num_columns, num_nonzeros, row_num, col_num;
    double nnz_val;
    char *nnz_val_str;
    char *line;
    FILE *fp;
    gsl_matrix *M;
    
    line = (char*)malloc(200*sizeof(char));
    fp = fopen(fname,"r");
    fgets(line,100,fp); //read comment
    fgets(line,100,fp); //read dimensions and nnzs 
    sscanf(line, "%d %d %d", &num_rows, &num_columns, &num_nonzeros);
    M = gsl_matrix_calloc(num_rows, num_columns); // calloc sets all elements to zero

    // read and set elements
    nnz_val_str = (char*)malloc(50*sizeof(char));
    for(i=0; i<num_nonzeros; i++){
        fgets(line,100,fp); 
        sscanf(line, "%d %d %s", &row_num, &col_num, nnz_val_str);
        nnz_val = atof(nnz_val_str);
        gsl_matrix_set(M, row_num, col_num, nnz_val);
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
gsl_matrix * matrix_load_from_binary_file(char *fname){
    int i, j, num_rows, num_columns, row_num, col_num;
    double nnz_val;
    size_t one = 1;
    FILE *fp;
    gsl_matrix *M;
    
    fp = fopen(fname,"r");
    fread(&num_rows,sizeof(int),one,fp); //read m
    fread(&num_columns,sizeof(int),one,fp); //read n
    printf("initializing M of size %d by %d\n", num_rows, num_columns);
    M = gsl_matrix_alloc(num_rows,num_columns);
    printf("done..\n");

    // read and set elements
    for(i=0; i<num_rows; i++){
        for(j=0; j<num_columns; j++){
            fread(&nnz_val,sizeof(double),one,fp); //read nnz
            gsl_matrix_set(M,i,j,nnz_val);
        }
    }
    fclose(fp);

    return M;
}


/* write matrix to binary file 
 * the nonzeros are in order of double loop over rows and columns
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
void matrix_write_to_binary_file(gsl_matrix *M, char *fname){
    int i, j, num_rows, num_columns, row_num, col_num;
    double nnz_val;
    size_t one = 1;
    FILE *fp;
    num_rows = M->size1; num_columns = M->size2;
    
    fp = fopen(fname,"w");
    fwrite(&num_rows,sizeof(int),one,fp); //write m
    fwrite(&num_columns,sizeof(int),one,fp); //write n

    // write the elements
    for(i=0; i<num_rows; i++){
        for(j=0; j<num_columns; j++){
            nnz_val = gsl_matrix_get(M,i,j);
            fwrite(&nnz_val,sizeof(double),one,fp); //write nnz
        }
    }
    fclose(fp);
}




/* load vector from file 
format:
% comment
num_rows
value
.....
value
*/
gsl_vector * vector_load_from_file(char *fname){
    int i, j, num_rows;
    double nnz_val;
    char *nnz_val_str;
    char *line;
    FILE *fp;
    gsl_vector *v;
    
    line = (char*)malloc(200*sizeof(char));
    fp = fopen(fname,"r");
    fgets(line,100,fp); //read comment
    fgets(line,100,fp); //read dimension 
    sscanf(line, "%d", &num_rows);
    v = gsl_vector_calloc(num_rows);

    // read and set elements
    nnz_val_str = (char*)malloc(50*sizeof(char));
    for(i=0; i<num_rows; i++){
        fgets(line,100,fp); 
        sscanf(line, "%s", nnz_val_str);
        nnz_val = atof(nnz_val_str);
        gsl_vector_set(v, i, nnz_val);
    }
    fclose(fp);

    // clean
    free(line);
    free(nnz_val_str);

    return v;
}


/* build up a random matrix R */
void initialize_random_matrix_old(gsl_matrix *M){
    int i,j,m,n;
    double nnz_val;
    m = M->size1;
    n = M->size2;

    // seed 
    srand(time(NULL));

    // read and set elements
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            gsl_matrix_set(M, i, j, ((double) rand() / (RAND_MAX)));
        }
    }
}


/* build up a random matrix R */
void initialize_random_matrix(gsl_matrix *M){
    int i,j,m,n;
    double nnz_val;
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set (r, time(NULL));

    m = M->size1;
    n = M->size2;

    // set random elements
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            gsl_matrix_set(M, i, j, gsl_rng_uniform (r));
        }
    }

    gsl_rng_free (r);
}



/*
% project v in direction of u
function p=project_vec(v,u)
p = (dot(v,u)/norm(u)^2)*u;
*/
void project_vector(gsl_vector *v, gsl_vector *u, gsl_vector *p){
    double dot_product_val, vec_norm, scalar_val; 
    gsl_blas_ddot(v, u, &dot_product_val);
    vec_norm = gsl_blas_dnrm2(u);
    scalar_val = dot_product_val/(vec_norm*vec_norm);
    gsl_vector_memcpy(p, u);
    gsl_vector_scale (p, scalar_val); 
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
void build_orthonormal_basis_from_mat(gsl_matrix *A, gsl_matrix *Q){
    int m,n,i,j,ind,num_ortos=2;
    double vec_norm;
    gsl_vector *vi,*vj,*p;
    m = A->size1;
    n = A->size2;
    vi = gsl_vector_calloc(m);
    vj = gsl_vector_calloc(m);
    p = gsl_vector_calloc(m);
    gsl_matrix_memcpy(Q, A);
    for(ind=0; ind<num_ortos; ind++){
        for(j=0; j<n; j++){
            gsl_matrix_get_col(vj, Q, j);
            for(i=0; i<j; i++){
                gsl_matrix_get_col(vi, Q, i);
                project_vector(vj, vi, p);
                gsl_vector_sub(vj, p);
            }
            vec_norm = gsl_blas_dnrm2(vj);
            gsl_vector_scale(vj, 1.0/vec_norm);
            gsl_matrix_set_col (Q, j, vj);
        }
    }
}


/* C = A*B */
void matrix_matrix_mult(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C){
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
}


/* C = A^T*B */
void matrix_transpose_matrix_mult(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C){
    gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
}


/* y = M*x */
void matrix_vector_mult(gsl_matrix *M, gsl_vector *x, gsl_vector *y){
    gsl_blas_dgemv (CblasNoTrans, 1.0, M, x, 0.0, y);
}


/* y = M^T*x */
void matrix_transpose_vector_mult(gsl_matrix *M, gsl_vector *x, gsl_vector *y){
    gsl_blas_dgemv (CblasTrans, 1.0, M, x, 0.0, y);
}


/* compute evals and evecs of symmetric matrix */
void compute_evals_and_evecs_of_symm_matrix(gsl_matrix *M, gsl_vector *eval, gsl_matrix *evec){
    gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc (M->size1);

    gsl_eigen_symmv (M, eval, evec, w);

    gsl_eigen_symmv_free(w);

    gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);
}


/* compute QR factorization 
M is mxn; Q is mxm and R is mxn
this is slow
*/
void compute_QR_factorization(gsl_matrix *M, gsl_matrix *Q, gsl_matrix *R){
    //printf("QR setup..\n");
    gsl_matrix *QR = gsl_matrix_calloc(M->size1, M->size2); 
    gsl_vector *tau = gsl_vector_alloc(min(M->size1,M->size2));
    gsl_matrix_memcpy (QR, M);

    //printf("QR decomp..\n");
    gsl_linalg_QR_decomp (QR, tau);
    //printf("QR unpack..\n");
    gsl_linalg_QR_unpack (QR, tau, Q, R);
    //printf("done QR..\n");
}



/* compute compact QR factorization 
M is mxn; Q is mxk and R is kxk
*/
void compute_QR_compact_factorization(gsl_matrix *M, gsl_matrix *Q, gsl_matrix *R){
    int i,j,m,n,k;
    m = M->size1;
    n = M->size2;
    k = min(m,n);

    //printf("QR setup..\n");
    gsl_matrix *QR = gsl_matrix_calloc(M->size1, M->size2); 
    gsl_vector *tau = gsl_vector_alloc(min(M->size1,M->size2));
    gsl_matrix_memcpy (QR, M);

    //printf("QR decomp..\n");
    gsl_linalg_QR_decomp (QR, tau);

    //printf("extract R..\n");
    for(i=0; i<k; i++){
        for(j=0; j<k; j++){
            if(j>=i){
                gsl_matrix_set(R,i,j,gsl_matrix_get(QR,i,j));
            }
        }
    }

    //printf("extract Q..\n");
    gsl_vector *vj = gsl_vector_calloc(m);
    for(j=0; j<k; j++){
        gsl_vector_set(vj,j,1.0);
        gsl_linalg_QR_Qvec (QR, tau, vj);
        gsl_matrix_set_col(Q,j,vj);
        vj = gsl_vector_calloc(m);
    } 
}



/* compute compact QR factorization and get Q 
M is mxn; Q is mxk and R is kxk (not computed)
*/
void QR_factorization_getQ(gsl_matrix *M, gsl_matrix *Q){
    int i,j,m,n,k;
    m = M->size1;
    n = M->size2;
    k = min(m,n);

    gsl_matrix *QR = gsl_matrix_calloc(M->size1, M->size2); 
    gsl_vector *tau = gsl_vector_alloc(min(M->size1,M->size2));
    gsl_matrix_memcpy (QR, M);

    gsl_linalg_QR_decomp (QR, tau);


    gsl_vector *vj = gsl_vector_calloc(m);
    for(j=0; j<k; j++){
        gsl_vector_set(vj,j,1.0);
        gsl_linalg_QR_Qvec (QR, tau, vj);
        gsl_matrix_set_col(Q,j,vj);
        vj = gsl_vector_calloc(m);
    } 

    gsl_vector_free(vj);
    gsl_vector_free(tau);
    gsl_matrix_free(QR);
}




/* build diagonal matrix from vector elements */
void build_diagonal_matrix(gsl_vector *dvals, int n, gsl_matrix *D){
    int i;
    for(i=0; i<n; i++){
        gsl_matrix_set(D,i,i,gsl_vector_get(dvals,i));
    }
}



/* invert diagonal matrix */
void invert_diagonal_matrix(gsl_matrix *Dinv, gsl_matrix *D){
    int i;
    for(i=0; i<(D->size1); i++){
        gsl_matrix_set(Dinv,i,i,1.0/(gsl_matrix_get(D,i,i)));
    }
}



/* frobenius norm */
double matrix_frobenius_norm(gsl_matrix *M){
    int i,j;
    double val, norm = 0;
    for(i=0; i<M->size1; i++){
        for(j=0; j<M->size2; j++){
            val = gsl_matrix_get(M, i, j);
            norm += val*val;
        }
    }
    norm = sqrt(norm);
    return norm;
}


/* print matrix */
void matrix_print(gsl_matrix *M){
    int i,j;
    double val;
    for(i=0; i<M->size1; i++){
        for(j=0; j<M->size2; j++){
            val = gsl_matrix_get(M, i, j);
            printf("%f  ", val);
        }
        printf("\n");
    }
}



/* P = U*S*V^T */
void form_svd_product_matrix(gsl_matrix *U, gsl_matrix *S, gsl_matrix *V, gsl_matrix *P){
    int k,m,n;
    m = P->size1;
    n = P->size2;
    k = S->size1;
    gsl_matrix * SVt = gsl_matrix_alloc(k,n);
    // form Svt = S*V^T
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, S, V, 0.0, SVt);
    // form P = U*S*V^T
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, U, SVt, 0.0, P);
}


/* calculate percent error between A and B: 100*norm(A - B)/norm(A) */
double get_percent_error_between_two_mats(gsl_matrix *A, gsl_matrix *B){
    int m,n;
    double normA, normB, normA_minus_B;
    m = A->size1;
    n = A->size2;
    gsl_matrix *A_minus_B = gsl_matrix_alloc(m,n);
    gsl_matrix_memcpy(A_minus_B, A);
    gsl_matrix_sub(A_minus_B,B);
    normA = matrix_frobenius_norm(A);
    normB = matrix_frobenius_norm(B);
    normA_minus_B = matrix_frobenius_norm(A_minus_B);
    return 100.0*normA_minus_B/normA;
}



/* copy the first k columns of M into M_out where k = M_out->ncols (M_out pre-initialized) */
void matrix_copy_first_columns(gsl_matrix *M_out, gsl_matrix *M){
    int i,k;
    k = M_out->size2;
    gsl_vector * col_vec;
    for(i=0; i<k; i++){
        col_vec = gsl_vector_calloc(M->size1);
        gsl_matrix_get_col(col_vec,M,i);
        gsl_matrix_set_col(M_out,i,col_vec);
        gsl_vector_free(col_vec);
    }
}


/* append matrices side by side: C = [A, B] */
void append_matrices_horizontally(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C){
    int i,j;

    for(i=0; i<A->size1; i++){
        for(j=0; j<A->size2; j++){
            gsl_matrix_set(C,i,j,gsl_matrix_get(A,i,j));
        }
    }

    for(i=0; i<B->size1; i++){
        for(j=0; j<B->size2; j++){
            gsl_matrix_set(C,i,A->size2 + j,gsl_matrix_get(B,i,j));
        }
    }
}



void estimate_rank_and_buildQ(gsl_matrix *M, double frac_of_max_rank, double TOL, gsl_matrix **Q, int *good_rank){
    int m,n,i,j,ind,maxdim;
    double vec_norm;
    gsl_matrix *RN,*Y,*Qbig,*Qsmall;
    gsl_vector *vi,*vj,*p,*p1;
    m = M->size1;
    n = M->size2;
    maxdim = round(min(m,n)*frac_of_max_rank);

    vi = gsl_vector_calloc(m);
    vj = gsl_vector_calloc(m);
    p = gsl_vector_calloc(m);
    p1 = gsl_vector_calloc(m);

    // build random matrix
    printf("form RN..\n");
    RN = gsl_matrix_calloc(n, maxdim);
    initialize_random_matrix(RN);

    // multiply to get matrix of random samples Y
    printf("form Y: %d x %d..\n",m,maxdim);
    Y = gsl_matrix_calloc(m, maxdim);
    matrix_matrix_mult(M, RN, Y);

    // estimate rank k and build Q from Y
    printf("form Qbig..\n");
    Qbig = gsl_matrix_calloc(m, maxdim);

    gsl_matrix_memcpy(Qbig, Y);

    printf("estimate rank with TOL = %f..\n", TOL);
    *good_rank = maxdim;
    int forbreak = 0;
    for(j=0; !forbreak && j<maxdim; j++){
        gsl_matrix_get_col(vj, Qbig, j);
        for(i=0; i<j; i++){
            gsl_matrix_get_col(vi, Qbig, i);
            project_vector(vj, vi, p);
            gsl_vector_sub(vj, p);
            if(gsl_blas_dnrm2(p) < TOL && gsl_blas_dnrm2(p1) < TOL){
                *good_rank = j;
                forbreak = 1;
                break;
            }
            gsl_vector_memcpy(p1,p);
        }
        vec_norm = gsl_blas_dnrm2(vj);
        gsl_vector_scale(vj, 1.0/vec_norm);
        gsl_matrix_set_col(Qbig, j, vj);
    }

    printf("estimated rank = %d\n", *good_rank);
    Qsmall = gsl_matrix_calloc(m, *good_rank);
    *Q = gsl_matrix_calloc(m, *good_rank);
    matrix_copy_first_columns(Qsmall, Qbig);
    QR_factorization_getQ(Qsmall, *Q);

    gsl_matrix_free(RN);
    gsl_matrix_free(Y);
    gsl_matrix_free(Qbig);
    gsl_matrix_free(Qsmall);
    gsl_vector_free(p);
    gsl_vector_free(p1);
    gsl_vector_free(vi);
    gsl_vector_free(vj);
}



void estimate_rank_and_buildQ2(gsl_matrix *M, int kblock, double TOL, gsl_matrix **Y, gsl_matrix **Q, int *good_rank){
    int m,n,i,j,ind,exit_loop = 0;
    double error_norm;
    gsl_matrix *RN,*Y_new,*Y_big,*QtM,*QQtM;
    gsl_vector *vi,*vj,*p,*p1;
    m = M->size1;
    n = M->size2;

    // build random matrix
    printf("form RN..\n");
    RN = gsl_matrix_calloc(n,kblock);
    initialize_random_matrix(RN);

    // multiply to get matrix of random samples Y
    printf("form Y: %d x %d..\n",m,kblock);
    *Y = gsl_matrix_calloc(m, kblock);
    matrix_matrix_mult(M, RN, *Y);

    ind = 0;
    while(!exit_loop){
        printf("form Q..\n");
        if(ind > 0){
            gsl_matrix_free(*Q);
        }
        *Q = gsl_matrix_calloc((*Y)->size1, (*Y)->size2);
        QR_factorization_getQ(*Y, *Q);

        // compute QtM
        QtM = gsl_matrix_calloc((*Q)->size2, M->size2);
        matrix_transpose_matrix_mult(*Q,M,QtM);

        // compute QQtM
        QQtM = gsl_matrix_calloc(M->size1, M->size2); 
        matrix_matrix_mult(*Q,QtM,QQtM);

        error_norm = 0.01*get_percent_error_between_two_mats(QQtM, M);

        printf("Y is of size %d x %d and error_norm = %f\n", (*Y)->size1, (*Y)->size2, error_norm);
        *good_rank = (*Y)->size2;
       
        // add more samples if needed
        if(error_norm > TOL){
            Y_new = gsl_matrix_calloc(m, kblock);
            initialize_random_matrix(RN);
            matrix_matrix_mult(M, RN, Y_new);

            Y_big = gsl_matrix_calloc((*Y)->size1, (*Y)->size2 + Y_new->size2); 
            append_matrices_horizontally(*Y, Y_new, Y_big);
            gsl_matrix_free(*Y);
            *Y = gsl_matrix_calloc(Y_big->size1,Y_big->size2);
            gsl_matrix_memcpy(*Y,Y_big);
            
            gsl_matrix_free(Y_new);
            gsl_matrix_free(Y_big);
            gsl_matrix_free(QtM);
            gsl_matrix_free(QQtM);
            ind++;
        }
        else{
            gsl_matrix_free(RN);
            exit_loop = 1;
        }    
    }
}

