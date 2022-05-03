/* matrix decomposition algorithms */
/* Sergey Voronin */

#include "rank_revealing_algorithms_intel_mkl.h"

/* computes the low rank SVD of rank k or tolerance TOL of matrix M  */
void low_rank_svd_decomp_fixed_rank_or_prec(mat *M, myint64 k, double TOL, myint64 *frank, mat **U, mat **S, mat **V){
    myint64 i,m,n,r,rankMode,tolMode;
    double sval;
    mat *Mc, *Uf, *Sf, *Vtf, *Vf;

    // get dims
    m = M->nrows;
    n = M->ncols; 
    r = min(m,n);

    // set up mats
    Mc = matrix_new(m,n);
    Uf = matrix_new(m,r);
    Sf = matrix_new(r,r);
    Vtf = matrix_new(r,n);
    Vf = matrix_new(n,r);

    // determine mode
    if(k <= 0){
        rankMode = 0;
        tolMode = 1;
        k = min(m,n);
    } else {
        rankMode = 1;
        tolMode = 0;
    }

    // copy matrix
    matrix_copy(Mc,M);

    // call SVD
    singular_value_decomposition(Mc, Uf, Sf, Vtf);
    matrix_delete(Mc);
    matrix_build_transpose(Vf, Vtf);
    //use_low_rank_svd_for_approximation(M, Uf, Sf, Vf);


    // set frank to k or to |\sigma_{k+1}| < TOL
    *frank = k;
    if(tolMode){
        for(i = 0; i<Sf->nrows; i++){
            sval = matrix_get_element(Sf,i,i);
            if(fabs(sval) < TOL && i<(Sf->nrows-1)){
                *frank = i+1;
                break;
            }
        }
    }

    // setup mats
    *U = matrix_new(m,*frank);
    *S = matrix_new(*frank,*frank);
    *V = matrix_new(n,*frank);

    // extract components
    fill_matrix_from_first_columns(Uf, *frank, *U);
    fill_matrix_from_first_columns(Vf, *frank, *V);
    fill_matrix_from_first_columns(Sf, *frank, *S);
    fill_matrix_from_first_rows(Sf, *frank, *S);

    matrix_delete(Uf); matrix_delete(Sf); matrix_delete(Vf);
}


/* computes the approximate low rank SVD of rank k of matrix M  */
void low_rank_svd_rand_decomp_fixed_rank(mat *M, myint64 k, myint64 p, myint64 vnum, myint64 q, myint64 s, myint64 *frank, mat **U, mat **S, mat **V){
    myint64 i,j,m,n,r,l;
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
    printf("get matrix of random samples..\n");
    Y = matrix_new(m, l);
    matrix_matrix_mult(M, RN, Y);
	matrix_delete(RN);

    // now build up (M M^T)^q R
    printf("power iteration..\n");
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

	// clean
	matrix_delete(Z);
	matrix_delete(Yorth);
	matrix_delete(Zorth);

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
    matrix_delete(Y);
    matrix_delete(Q);
    //matrix_delete(Z);
    //matrix_delete(Yorth);
    //matrix_delete(Zorth);
}



/* column pivoted QR for mxn A 
input: matrix M
outputs: matrices Q and R and vector I such that A(:,I) = QR */
void pivotedQR_mkl(mat *M, mat **Q, mat **R, vec **I){
    myint64 i,j,k,m,n,Rrows,Rcols,Qrows,Qcols;
    mat *Mwork;
    vec *col_vec;
    m = M->nrows; n = M->ncols;
    k = min(m,n);
    
    Mwork = matrix_new(m,n);
    matrix_copy(Mwork,M);

    int *Iarr = (int*)malloc(n*sizeof(int));
    double *tau_arr = (double*)malloc(min(m,n)*sizeof(double));

    // set up dimensions
    if(m <= n){
        Qrows = k; Qcols = k;
        Rrows = k; Rcols = n;
    } else {
        Qrows = m; Qcols = k;
        Rrows = k; Rcols = k;
    }
    
    // get R
    LAPACKE_dgeqp3(CblasColMajor, Mwork->nrows, Mwork->ncols, (double*)Mwork->d, Mwork->nrows, Iarr, tau_arr);

    *R = matrix_new(Rrows,Rcols);
    for(i=0; i<Rrows; i++){
        for(j=i; j<Rcols; j++){
            matrix_set_element(*R,i,j,matrix_get_element(Mwork,i,j));
        }
    }

    // get Q
    LAPACKE_dorgqr(CblasColMajor, Qrows, Qcols, k, (double*)Mwork->d, m, tau_arr);

    *Q = matrix_new(Qrows,Qcols);
    for(i=0; i<Qrows; i++){
        for(j=0; j<Qcols; j++){
            matrix_set_element(*Q,i,j,matrix_get_element(Mwork,i,j));
        }
    }

    // get I
    *I = vector_new(n);
    for(i=0; i<n; i++){
        vector_set_element(*I,i,(myint64)(Iarr[i]-1));
    }

    // free temp variables
    matrix_delete(Mwork);
    free(Iarr);
    free(tau_arr);
}



/* generates the householder vector as a matrix for use in the 
partial column pivoted QR routine */
void get_householder_matrix(vec *x, myint64 ind1, myint64 ind2, mat *H){
    myint64 i,j;
    double val,normval;

    for(i=ind1; i<ind2; i++){
        matrix_set_element(H,i,0,vector_get_element(x,i));
    }

    normval = 0;
    for(i=ind1; i<ind2; i++){
        val = vector_get_element(x,i);
        normval += val*val;
    }
    normval = sqrt(normval);

    matrix_set_element(H,ind1,0,matrix_get_element(H,ind1,0) - normval);


    val = get_matrix_frobenius_norm(H);
    if(val > 0){
        matrix_scale(H, sqrt(2/(val*val)));
    }
}



/* evaluate approximation to M using supplied low rank SVD of rank k */
void use_low_rank_svd_for_approximation(mat *M, mat *U, mat *S, mat *V){
    mat *P;
    P = matrix_new(M->nrows, M->ncols);
    form_svd_product_matrix(U, S, V, P);

    printf("norm(M,fro) = %f\n", get_matrix_frobenius_norm(M));
    printf("norm(P,fro) = %f\n", get_matrix_frobenius_norm(P));
    printf("percent error = %f\n", get_percent_error_between_two_mats(M,P));

    matrix_delete(P);
}


/* evaluate approximation to M using supplied partial pivoted QR decomposition */
void use_pivoted_QR_decomp_for_approximation(mat *M, mat *Qk, mat *Rk, vec *I){
    myint64 i,m,n;
    double percent_error;
    mat *P, *Porig, *QkRk, *QkRkPt;
    vec *col_vec;
    m = M->nrows; n = M->ncols;

    QkRk = matrix_new(m,n);
    QkRkPt = matrix_new(m,n);
    Porig = matrix_new(n,n);
    P = matrix_new(n,n);

    printf("building P..\n");
    initialize_identity_matrix(Porig);
    #pragma omp parallel shared(Porig,P,I) private(i,col_vec) 
    {
    #pragma omp for
    for(i=0; i<Porig->ncols; i++){
        col_vec = vector_new(Porig->ncols);
        matrix_get_col(Porig,vector_get_element(I,i),col_vec);
        matrix_set_col(P,i,col_vec);
        vector_delete(col_vec);
    }
    }
    

    matrix_matrix_mult(Qk,Rk,QkRk);
    matrix_matrix_transpose_mult(QkRk,P,QkRkPt);
    percent_error = get_percent_error_between_two_mats(M,QkRkPt);
    printf("percent_error between M and QkRkPt = %f\n", percent_error);    
    
    matrix_delete(Qk); matrix_delete(Rk);
    matrix_delete(QkRk); matrix_delete(QkRkPt);
    matrix_delete(Porig); matrix_delete(P);
}

