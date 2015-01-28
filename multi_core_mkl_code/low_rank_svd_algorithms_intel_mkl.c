#include "low_rank_svd_algorithms_intel_mkl.h"


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



/* computes the approximate low rank SVD of rank k of matrix M using QR version 
 * with range sampling via (M M^T)^q M R*/
void randomized_low_rank_svd3_old(mat *M, int k, int q, mat **U, mat **S, mat **V){
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



/* computes the approximate low rank SVD of rank k of matrix M using QR version 
 * with range sampling via (M M^T)^q M R*/
void randomized_low_rank_svd3(mat *M, int k, int q, int s, mat **U, mat **S, mat **V){
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

    // now build up (M M^T)^q R
    mat *Z = matrix_new(n,k);
    mat *Yorth;
    mat *Zorth;
    Yorth = matrix_new(m,k);
    Zorth = matrix_new(n,k);
    for(j=1; j<q; j++){
        printf("in loop for j=%d of %d\n", j, q);

        if((2*j-2) % s == 0){
            printf("orthogonalize Y..\n");
            QR_factorization_getQ(Y, Yorth);
            printf("Z = M'*Yorth..\n");
            matrix_transpose_matrix_mult(M,Yorth,Z);
        }
        else{
            printf("Z = M'*Y..\n");
            matrix_transpose_matrix_mult(M,Y,Z);
        }

        
        if((2*j-1) % s == 0){
            printf("orthogonalize Z..\n");
            QR_factorization_getQ(Z, Zorth);
            printf("Y = M*Zorth..\n");
            matrix_matrix_mult(M,Zorth,Y);
        }
        else{
            printf("Y = M*Z..\n");
            matrix_matrix_mult(M,Z,Y);
        }
    }

    // orthogonalize on exit from loop to get Q
    mat *Q = matrix_new(m,k);
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
    matrix_delete(Z);
    matrix_delete(Rhat);
    matrix_delete(Qhat);
    matrix_delete(Uhat);
    matrix_delete(Vhat_trans);
    matrix_delete(Bt);
    matrix_delete(Yorth);
    matrix_delete(Zorth);
}


/* computes the approximate low rank SVD of rank k of matrix M using QR version 
automatically estimates the rank needed */
void randomized_low_rank_svd2_autorank1(mat *M, double frac_of_max_rank, double TOL, mat **U, mat **S, mat **V){
    int i,j,m,n,k,kinit;
    double val;
    mat *Q;
    m = M->nrows; n = M->ncols;
    kinit = min(m,n);

    // estimate rank k and build Q from Y
    printf("estimating rank and building Q..\n");
    //build_orthonormal_basis_from_mat(Y,Q);
    //QR_factorization_getQ(Y, Q);
    estimate_rank_and_buildQ(M,frac_of_max_rank,TOL,&Q,&k);
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
automatically estimates the rank needed */
void randomized_low_rank_svd2_autorank2(mat *M, int kblocksize, double TOL, mat **U, mat **S, mat **V){
    int i,j,m,n,k;
    double val;
    mat *Q;
    m = M->nrows; n = M->ncols;

    // estimate rank k and build Q from Y
    printf("estimating rank and building Q..\n");
    //estimate_rank_and_buildQ(M,frac_of_max_rank,TOL,&Q,&k);
    estimate_rank_and_buildQ2(M, kblocksize, TOL, &Q, &k);
    printf("estimated rank = %d\n", k);
    printf("norm(Q,fro) = %f\n", get_matrix_frobenius_norm(Q));

    // setup U, S, and V 
    *U = matrix_new(m,k);
    *S = matrix_new(k,k);
    *V = matrix_new(n,k);

    // form Bt = Mt*Q : nxm * mxk = nxk
    printf("form Bt..\n");
    mat *Bt = matrix_new(n,k);
    printf("M dims: %d x %d\n", M->nrows, M->ncols);
    printf("Q dims: %d x %d\n", Q->nrows, Q->ncols);
    printf("Bt dims: %d x %d\n", Bt->nrows, Bt->ncols);
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

