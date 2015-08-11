#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"
#include "mkl_lapacke.h"
#include "matrix.h"
#include "mex.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))

void mexFunction( int nlhs, mxArray *plhs[],
             int nrhs, const mxArray *prhs[] ){
    int i, j, m, n, k, info, solve_type;
    time_t start_time, end_time;
    double  *pr;
    lapack_int *Iarr;
    double *tauarr,*data;

    /* Check for proper number of arguments. */
    if (nrhs != 1) 
        mexErrMsgTxt("One input required, the matrix.\n");

    srand(time(NULL));

    /* get input parameters */
    n = mxGetN(prhs[0]);
    m = mxGetM(prhs[0]);
    pr = mxGetPr(prhs[0]);
    
    // can use mxCalloc here too..
    data = (double*)calloc(m*n,sizeof(double));
    
    for(i=0; i<(m*n); i++){
        data[i] = ((double) rand() / (RAND_MAX));
    }

    printf("Starting up MKL..\n");

    printf("call QR\n");
    Iarr = (lapack_int*)calloc(n,sizeof(lapack_int));
    tauarr = (double*)calloc(MIN(m,n),sizeof(double));
    
    // call LAPACKE_dgeqp3 on matlab data
    info=LAPACKE_dgeqp3(LAPACK_COL_MAJOR, m, n, pr, m, Iarr, tauarr);
    
    //printf("After first call to QR, info = %d\n",info);
    /*for(i=0; i<m; i++){
       printf("\n%d %f ",Iarr[i],tauarr[i]); 
       for(j=0; j<n; j++)
         printf(" %f",pr[i+j*m]);
    }*/
    
     
    // call LAPACKE_dgeqp3 on random data
    for(i=0; i<n; i++)Iarr[i]=0;
    info=LAPACKE_dgeqp3(LAPACK_COL_MAJOR, m, n, data, m, Iarr, tauarr);

    //printf("\n\nAfter second call to QR, info = %d\n",info);
    /*for(i=0; i<m; i++){
       printf("\n%d %f ",Iarr[i],tauarr[i]); 
       for(j=0; j<n; j++)
            printf(" %f",data[i+j*m]);
       }*/

    //printf("\nAbout to free data\n");
    free(data);
    free(Iarr);
    free(tauarr); 
    return;
}

