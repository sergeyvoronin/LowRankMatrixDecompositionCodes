#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"
#include "mkl_lapacke.h"
#include "matrix.h"
#include "mex.h"

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

#include "rank_revealing_algorithms_intel_mkl.h"


/* mex function to compute rsvd given matrix A and rank k, return matrices U, S, V */
void mexFunction( int nlhs, mxArray *plhs[],
             int nrhs, const mxArray *prhs[] ){
    int i, j, m, n, k, info, verbose = 0;
    time_t start_time, end_time;
    double  *ptr;
    double normM,normU,normS,normV,normP,percent_error;
    mat *M, *T;
    vec *I; 
    char error_string[50];
    
    mwSize dims[2];
    size_t bytes_to_copy;

    /* Check for proper number of arguments. */
    if (nrhs != 2){ 
        sprintf(error_string, "Two inputs required, the matrix and rank k.\n");
        mexErrMsgTxt(error_string);
    }

    srand(time(NULL));

    /* get input parameters */
    n = mxGetN(prhs[0]);
    m = mxGetM(prhs[0]);
    ptr = mxGetPr(prhs[0]);
    k = *mxGetPr(prhs[1]);

    if (k > min(m,n) ){
        sprintf(error_string, "k must be less than min(m,n) = %d\n", min(m,n));
        mexErrMsgTxt(error_string);
    }

    /* setup matrix from matlab data */
    if(verbose)
        printf("the matrix is of size %d x %d\n", m, n);
    M = matrix_new(m,n);
    for(i=0; i<(m*n); i++){
        M->d[i] = ptr[i];
    }
    
    // call one of the algorithms here
    if(verbose)
        printf("calling ID with k =%d\n", k);
    id_decomp_fixed_rank(M, k, &I, &T);

    // adjust indices of I by 1 for Matlab notation
    for(i=0; i<n; i++){
        I->d[i] = I->d[i] + 1;
    }

    // create I,T in Matlab
    if(verbose)
        printf("creating vector I: %d by %d..\n", I->nrows, 1);
    dims[0] = I->nrows; dims[1] = 1;
    plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    
    ptr = mxGetPr(plhs[0]);
    bytes_to_copy = dims[0]*dims[1]*sizeof(double);
    memcpy(ptr,I->d,bytes_to_copy);
    

    printf("creating T: %d by %d..\n", T->nrows, T->ncols);
    dims[0] = T->nrows; dims[1] = T->ncols;
    plhs[1] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    
    ptr = mxGetPr(plhs[1]);
    bytes_to_copy = dims[0]*dims[1]*sizeof(double);
    memcpy(ptr,T->d,bytes_to_copy);

    // delete and exit
    matrix_delete(M);
    matrix_delete(T);
    vector_delete(I);

    return;
}

