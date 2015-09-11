#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "mex.h"

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

#include "rank_revealing_algorithms_nvidia_cula.h"


/* mex function to compute rsvd given matrix A and rank k, return matrices U, S, V */
void mexFunction( int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[] ){
    int i, j, m, n, k, q, kstep, nstep, info, verbose = 0;
    time_t start_time, end_time;
    double  *ptr;
    double normM,normU,normS,normV,normP,percent_error;
    mat *M, *U, *S, *V, *P;
    char error_string[50];
    
    mwSize dims[2];
    size_t bytes_to_copy;
    culaStatus status;

    /* Check for proper number of arguments. */
    if (nrhs != 4){ 
        sprintf(error_string, "Three inputs required, the matrix, kstep (block size), nstep (number of blocks), q (power scheme parameter).\n");
        mexErrMsgTxt(error_string);
    }

    printf("Initializing CULA\n");
    status = culaInitialize();
    checkStatus(status);

    srand(time(NULL));

    /* get input parameters */
    n = mxGetN(prhs[0]);
    m = mxGetM(prhs[0]);
    ptr = mxGetPr(prhs[0]);
    kstep = *mxGetPr(prhs[1]);
    nstep = *mxGetPr(prhs[2]);
    q = *mxGetPr(prhs[3]);

    if (kstep*nstep > min(m,n) ){
        sprintf(error_string, "kstep*nstep must be less than min(m,n) = %d\n", min(m,n));
        mexErrMsgTxt(error_string);
    }


    /* setup matrix from matlab data */
    if(verbose)
        printf("the matrix is of size %d x %d\n", m, n);
    M = matrix_new(m,n);
    for(i=0; i<(m*n); i++){
        M->d[i] = ptr[i];
    }
    //matrix_print(M);

    // call one of the algorithms here
    if(verbose)
        printf("calling random SVD with kstep =%d, nstep = %d\n", kstep, nstep);
    randomized_low_rank_svd4(M, kstep, nstep, q, &U, &S, &V);

    // print stats if needed
    if(verbose){
        P = matrix_new(m,n);
        form_svd_product_matrix(U,S,V,P);
        normM = get_matrix_frobenius_norm(M);
        normU = get_matrix_frobenius_norm(U);
        normS = get_matrix_frobenius_norm(S);
        normV = get_matrix_frobenius_norm(V);
        normP = get_matrix_frobenius_norm(P);
        printf("normM = %f ; normU = %f ; normS = %f ; normV = %f ; normP = %f\n", normM, normU, normS, normV, normP);
        percent_error = get_percent_error_between_two_mats(M,P);
        printf("percent_error between M and U S V^T = %f\n", percent_error);
    }

    // create U,S,V in Matlab
    printf("creating U: %d by %d..\n", U->nrows, U->ncols);
    dims[0] = U->nrows; dims[1] = U->ncols;
    plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    
    ptr = mxGetPr(plhs[0]);
    bytes_to_copy = dims[0]*dims[1]*sizeof(double);
    memcpy(ptr,U->d,bytes_to_copy);
    

    printf("creating S: %d by %d..\n", S->nrows, S->ncols);
    dims[0] = S->nrows; dims[1] = S->ncols;
    plhs[1] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    
    ptr = mxGetPr(plhs[1]);
    bytes_to_copy = dims[0]*dims[1]*sizeof(double);
    memcpy(ptr,S->d,bytes_to_copy);

    printf("creating V: %d by %d..\n", V->nrows, V->ncols);
    dims[0] = V->nrows; dims[1] = V->ncols;
    plhs[2] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    
    ptr = mxGetPr(plhs[2]);
    bytes_to_copy = dims[0]*dims[1]*sizeof(double);
    memcpy(ptr,V->d,bytes_to_copy);

    // delete and exit
    matrix_delete(M);
    matrix_delete(U);
    matrix_delete(S);
    matrix_delete(V);

    if(verbose){
        matrix_delete(P);
    }

    printf("Shutting down CULA\n");
    culaShutdown();

    return;
}

