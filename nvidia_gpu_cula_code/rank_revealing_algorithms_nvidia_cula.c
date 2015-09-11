#include "rank_revealing_algorithms_nvidia_cula.h"


/* computes the approximate low rank SVD of rank k of matrix M using BBt version */
void randomized_low_rank_svd1(mat *M, int k, mat **U, mat **S, mat **V){
    int i,j,m,n;
    double val;
    m = M->nrows; n = M->ncols;

    printf("running randomized_low_rank_svd1 with k = %d\n", k);

    // setup U, S, and V 
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
    printf("form Q..\n");
    mat *Q = matrix_new(m,k);
    //build_orthonormal_basis_from_mat(Y,Q);
    QR_factorization_getQ(Y, Q);


    // build the matrix B B^T = Q^T M M^T Q column by column 
    // Bt = M^T Q ; nxm * mxk = nxk
    printf("form BBt..\n");
    mat *B = matrix_new(k,n);
    matrix_transpose_matrix_mult(Q,M,B);

    mat *BBt = matrix_new(k,k);
    matrix_matrix_transpose_mult(B,B,BBt);    

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
    matrix_transpose_matrix_mult(B,UhatSinv,*V);

    // clean up
    matrix_delete(RN);
    matrix_delete(Y);
    matrix_delete(Q);
    matrix_delete(B);
    matrix_delete(Uhat);
    matrix_delete(Sinv);
    matrix_delete(UhatSinv);
    vector_delete(singvals);
    vector_delete(evals);
}



/* computes the approximate low rank SVD of rank k of matrix M using QR version */
void randomized_low_rank_svd2(mat *M, int k, mat **U, mat **S, mat **V){
    int i,j,m,n;
    double val;
    m = M->nrows; n = M->ncols;

    printf("running randomized_low_rank_svd2 with k = %d\n", k);

    // setup U, S, and V 
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

    // setup U, S, and V 
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
    matrix_delete(W);
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

    printf("running randomized_low_rank_svd3 with k = %d, q = %d, s = %d\n", k,q,s);

    // setup U, S, and V 
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
        printf("in loop for j=%d of %d\n", j, q-1);

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


/* computes the approximate low rank SVD of rank k of matrix M using the 
QB blocked algorithm for Q and BBt method */
void randomized_low_rank_svd4(mat *M, int kstep, int nstep, int q, mat **U, mat **S, mat **V){
    int i,j,m,n,knew;
    double val;
    m = M->nrows; n = M->ncols;
    mat *Q, *B, *BBt;

    printf("running randomized_low_rank_svd4 with kstep = %d, nstep = %d, q = %d\n", kstep, nstep, q);

    
    // setup mats
    knew = kstep*nstep;
    *U = matrix_new(m,knew);
    *S = matrix_new(knew,knew);
    *V = matrix_new(n,knew);

    printf("calling randQB with kstep = %d and nstep = %d and q = %d\n", kstep, nstep, q);
    randQB_pb(M, kstep, nstep, q, &Q, &B);

    BBt = matrix_new(knew,knew);
    matrix_matrix_transpose_mult(B,B,BBt); 

    // compute eigendecomposition of BBt
    printf("eigendecompose BBt..\n");
    vec *evals = vector_new(knew);
    mat *Uhat = matrix_new(knew, knew);
    matrix_copy_symmetric(Uhat,BBt);
    compute_evals_and_evecs_of_symm_matrix(Uhat, evals);

    // compute singular values and matrix Sigma
    printf("form S..\n");
    vec *singvals = vector_new(knew);
    for(i=0; i<knew; i++){
        vector_set_element(singvals,i,sqrt(vector_get_element(evals,i)));
    }
    initialize_diagonal_matrix(*S, singvals);
    
    // compute U = Q*Uhat mxk * kxk = mxk  
    printf("form U..\n");
    matrix_matrix_mult(Q,Uhat,*U);

    // compute nxk V 
    // V = B^T Uhat * Sigma^{-1}
    printf("form V..\n");
    mat *Sinv = matrix_new(knew,knew);
    mat *UhatSinv = matrix_new(knew,knew);
    invert_diagonal_matrix(Sinv,*S);
    matrix_matrix_mult(Uhat,Sinv,UhatSinv);
    matrix_transpose_matrix_mult(B,UhatSinv,*V);

    // clean up
    printf("clean up..\n");
    matrix_delete(Q);
    matrix_delete(B);
    matrix_delete(BBt);
    matrix_delete(Uhat);
    matrix_delete(Sinv);
    matrix_delete(UhatSinv);
    vector_delete(singvals);
    vector_delete(evals);
}


/* computes the approximate low rank SVD of rank k of matrix M using QR version 
automatically estimates the rank needed */
void randomized_low_rank_svd2_autorank1(mat *M, double frac_of_max_rank, double TOL, mat **U, mat **S, mat **V){
    int i,j,m,n,k,kinit;
    double val;
    mat *Q;
    m = M->nrows; n = M->ncols;
    kinit = min(m,n);

    printf("running randomized_low_rank_svd2_autorank1 with frac_of_max_rank = %f, TOL = %f\n", frac_of_max_rank, TOL);

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
    mat *Y,*Q;
    m = M->nrows; n = M->ncols;

    printf("running randomized_low_rank_svd2_autorank2 with kblocksize = %d, TOL = %f\n", kblocksize, TOL);

    // estimate rank k and build Q from Y
    printf("estimating rank and building Q..\n");
    estimate_rank_and_buildQ2(M, kblocksize, TOL, &Y, &Q, &k);
    printf("estimated rank = %d\n", k);
    //printf("norm(Q,fro) = %f\n", get_matrix_frobenius_norm(Q));

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



/* computes the approximate low rank SVD of rank k of matrix M using QR version 
via (M M^T)^q M R, automatically estimates the rank needed */
void randomized_low_rank_svd3_autorank2(mat *M, int kblocksize, double TOL, int q, int s, mat **U, mat **S, mat **V){
    int i,j,m,n,k;
    double val;
    mat *Y,*Q;
    m = M->nrows; n = M->ncols;

    printf("running randomized_low_rank_svd3_autorank2 with kblocksize = %d, TOL = %f, q = %d, s = %d\n", kblocksize,TOL,q,s);

    printf("estimating rank..\n");
    estimate_rank_and_buildQ2(M, kblocksize, TOL, &Y, &Q, &k);
    printf("estimated rank = %d\n", k);

    // setup mats
    *U = matrix_new(m,k);
    *S = matrix_new(k,k);
    *V = matrix_new(n,k);

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
    //mat *Q = matrix_new(m,k);
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


/* generates the householder vector as a matrix for use in the 
partial column pivoted QR routine */
void get_householder_matrix(vec *x, int ind1, int ind2, mat *H){
    int i,j;
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


/* computes the partial pivoted QR factorization of speicifed rank 
: A(:,I) \approx Qk*Rk 
inputs: matrix M and rank k
outputs: returned rank frank, matrices Qk, Rj, index vector I
*/
void pivoted_QR_of_specified_rank(mat *M, int k, int *frank, mat **Qk, mat **Rk, vec **I){
    int i,j,ind,q,m,n,max_colnorm_index,column_norm_zero,loop_lim,checkQRresult;
    double max_colnorm,tempval,val1,val2,val3;
    int * max_colnorm_indices; 
    mat *Q, *R, *Rk1, *Rk2, *invRk1, *H, *HHt, *HtR, *HHtR, *QHHt, *T;
    mat *QH;
    vec *column_norms1, *column_norms2, *row_vec1, *row_vec2, *col_vec1, *col_vec2, *Ientries;
    m = M->nrows;
    n = M->ncols;

    // set up mats
    Q = matrix_new(m,m);
    R = matrix_new(m,n);
    H = matrix_new(m,1);
    QHHt = matrix_new(m,m);
    HtR = matrix_new(1,n);
    HHtR = matrix_new(m,n);
    QH = matrix_new(m,1);

    matrix_copy(R,M);
    
    initialize_identity_matrix(Q);
    
    column_norms1 = vector_new(n);
    column_norms2 = vector_new(n);
    compute_matrix_column_norms(M,column_norms1); 

    // loop parameters
    column_norm_zero = 0;
    *frank = 0;

    //construct vector I
    *I = vector_new(n);
    for(i=0; i<n; i++){
        vector_set_element(*I,i,i);
    }

    for(i=0; !column_norm_zero && i<k; i++){
        
        if( i%50 == 0 ){
            printf("=========> iteration %d of %d--->\n", i+1, k);
        }
        
        //printf("find max_colnorm..\n");
        max_colnorm = vector_get_element(column_norms1,i);
        max_colnorm_index = i;
        for(j=(i+1); j<n; j++){
            val1 = vector_get_element(column_norms1,j);
            if(val1 > max_colnorm){
                max_colnorm_index = j;
                max_colnorm = val1;   
            }
        }


        /* break; Qk and Rk already set for i>0 */
        if(vector_get_element(column_norms1,max_colnorm_index) == 0 && i>0){
            column_norm_zero = 1;
            printf("column norm zero detected and will break!\n");
            break;
        }
        else{
            *frank = i+1;
        }
       

        // swap I
        val1 = vector_get_element(*I,i);
        vector_set_element(*I,i,vector_get_element(*I,max_colnorm_index));
        vector_set_element(*I,max_colnorm_index,val1);


        // swap R columns
        col_vec1 = vector_new(R->nrows);
        col_vec2 = vector_new(R->nrows);
        matrix_get_col(R,i,col_vec1);
        matrix_get_col(R,max_colnorm_index,col_vec2);
        matrix_set_col(R,i,col_vec2);
        matrix_set_col(R,max_colnorm_index,col_vec1); 
        vector_delete(col_vec1); 
        vector_delete(col_vec2); 


        // swap column norms
        val1 = vector_get_element(column_norms1, max_colnorm_index);
        vector_set_element(column_norms1,max_colnorm_index,vector_get_element(column_norms1,i));
        vector_set_element(column_norms1,i,val1);


        // get H and transform R and Q
        col_vec1 = vector_new(R->nrows);
        matrix_get_col(R,i,col_vec1);
        matrix_scale(H,0);
        get_householder_matrix(col_vec1, i, m, H);
        vector_delete(col_vec1); 

        matrix_transpose_matrix_mult(H,R,HtR);
        matrix_matrix_mult(H,HtR,HHtR);
        matrix_sub(R, HHtR);

        matrix_matrix_mult(Q,H,QH);
        matrix_matrix_transpose_mult(QH,H,QHHt);
        matrix_sub(Q,QHHt);

        //downgrade norms
        if(i != (n-1)){
            #pragma omp parallel shared(column_norms1,R,i) private(ind,val1,val2) 
            {
            #pragma omp for
            for(ind = i+1; ind < n ; ind++){
                val1 = vector_get_element(column_norms1,ind);
                val2 = matrix_get_element(R,i,ind); 
                vector_set_element(column_norms1,ind,val1 - val2*val2);
            } 
            }
        }
    }        

    //construct Qk and Rk
    *Qk = matrix_new(m,*frank);
    *Rk = matrix_new(*frank,n);

    fill_matrix_from_first_columns(Q, *frank, *Qk);
    fill_matrix_from_first_rows(R, *frank, *Rk);

    
    // delete temp variables
    matrix_delete(Q);
    matrix_delete(R);
    matrix_delete(H);
    matrix_delete(QHHt);
    matrix_delete(HtR);
    matrix_delete(HHtR);
    matrix_delete(QH);
    vector_delete(column_norms1);
    vector_delete(column_norms2);
}


/* randQB single vector algorithm with power method 
inputs: matrix M, rank k [integer (>0)], power scheme parameter p [ integer (>=0) ]
outputs: matrices Q and B s.t. M \approx Q*B 
*/
void randQB_p(mat *M, int k, int p, mat **Q, mat **B){
    int i,j,m,n;
    double dotp;
    mat *A,*RN;
    vec *ej,*rj,*pj,*qj,*qi,*yj,*bj;

    m = M->nrows; n = M->ncols;

    // build random matrix
    printf("form RN..\n");
    RN = matrix_new(n, k);
    initialize_random_matrix(RN);

    // set up mats
    A = matrix_new(m,n);
    *Q = matrix_new(m,k);
    *B = matrix_new(k,n);

    ej = vector_new(k);
    rj = vector_new(n);
    pj = vector_new(n);
    yj = vector_new(m);
    qj = vector_new(m);
    qi = vector_new(m);
    bj = vector_new(n);


    // copy M to A
    matrix_copy(A,M);

    for(j=0; j<k; j++){
        if(j%20 == 0){
            printf("in iteration %d\n", j);
        }
        vector_scale(ej,0);
        vector_set_element(ej,j,1); 
        matrix_vector_mult(RN,ej,rj);
        matrix_vector_mult(A,rj,yj);

        // power method
        for(i=0; i<p; i++){
            matrix_transpose_vector_mult(A,yj,pj);
            matrix_vector_mult(A,pj,yj);
        }

        // yj = yj - Q(:,i)*dot(Q(:,i),yj);
        for(i=0; i<(j-1); i++){
            matrix_get_col(*Q, i, qi);
            dotp = vector_dot_product(qi,yj);
            vector_scale(qi,dotp);
            vector_sub(yj,qi);
        }
        vector_copy(qj,yj);
        vector_scale(qj,1.0/vector_get2norm(qj));

        matrix_set_col(*Q, j, qj);

        matrix_transpose_vector_mult(A,qj,bj);

        matrix_set_row(*B, j, bj);

        // A = A - qj*bj;
        matrix_sub_column_times_row_vector(A,qj,bj);
    }

    // clean up
    matrix_delete(A);
    matrix_delete(RN);
    vector_delete(ej);
    vector_delete(rj);
    vector_delete(yj);
    vector_delete(pj);
    vector_delete(qj);
    vector_delete(qi);
    vector_delete(bj);
}


/* randQB blocked algorithm with power method 
randQB single vector algorithm with power method 
inputs: matrix M, integer kstep (block size), integer nstep (number of blocks), power scheme parameter p [ integer (>=0) ]
outputs: matrices Q and B s.t. M \approx Q*B 
*/
void randQB_pb(mat *M, int kstep, int nstep, int p, mat **Q, mat **B){
    int i,j,m,n,l,s;
    double dotp,elapsed_time;
    int *inds_local, *inds_global;
    mat *A, *RN, *RNp, *Yp, *Qp, *Bp, *AtQp, *AtQp2, *QpBp, *Qj, *QjtQp, *QjQjtQp;
    vec *ej,*rj,*pj,*qj,*qi,*yj,*bj;
    struct timeval start_timeval, end_timeval;

    m = M->nrows; n = M->ncols;
    l = kstep*nstep;

    // build random matrix
    printf("form RN..\n");
    RN = matrix_new(n, l);
    initialize_random_matrix(RN);

    // set up mats
    A = matrix_new(m,n);
    *Q = matrix_new(m,l);
    *B = matrix_new(l,n);
    RNp = matrix_new(n,kstep); 
    Yp = matrix_new(m,kstep);
    Qp = matrix_new(m,kstep);
    Bp = matrix_new(kstep,n);
    QpBp = matrix_new(m,n);
    AtQp = matrix_new(n,kstep);
    AtQp2 = matrix_new(n,kstep);


    // copy M to A
    matrix_copy(A,M);

    for(s=0; s<nstep; s++){
        printf("in step %d\n", s);

        
        gettimeofday(&start_timeval, NULL);
        inds_local = (int*)malloc(kstep*sizeof(int));
        for(i=0; i<kstep; i++){
            inds_local[i] = kstep*s + i;
        } 
        matrix_get_selected_columns(RN, inds_local, RNp);
        matrix_matrix_mult(A,RNp,Yp);
        gettimeofday(&end_timeval, NULL);
        elapsed_time = get_seconds_frac(start_timeval,end_timeval);
        //printf("elapsed time for building Yp: %4.8f sec\n", elapsed_time);


        // power method
        gettimeofday(&start_timeval, NULL);
        for(i=0; i<p; i++){
            QR_factorization_getQ(Yp, Qp);
            matrix_transpose_matrix_mult(A,Qp,AtQp);
            QR_factorization_getQ(AtQp, AtQp2);
            matrix_matrix_mult(A,AtQp2,Yp);
        }

        // Qp = qr(Yp,0) 
        QR_factorization_getQ(Yp, Qp);
        gettimeofday(&end_timeval, NULL);
        elapsed_time = get_seconds_frac(start_timeval,end_timeval);
        //printf("elapsed time power method: %4.8f sec\n", elapsed_time);


        // project Qp away from previous Q stuff
        if(s>0){
            gettimeofday(&start_timeval, NULL);
            inds_global = (int*)malloc(s*kstep*sizeof(int));
            Qj = matrix_new(m,s*kstep);
            QjtQp = matrix_new(s*kstep,kstep);
            QjQjtQp = matrix_new(m,kstep);

            for(i=0; i<(s*kstep); i++){
                inds_global[i] = i;
            }
            matrix_get_selected_columns(*Q, inds_global, Qj);

            // Yp = Qp - Q(:,J)*(Q(:,J)'*Qp);
            matrix_transpose_matrix_mult(Qj,Qp,QjtQp);
            matrix_matrix_mult(Qj,QjtQp,QjQjtQp);
            matrix_copy(Yp,Qp);
            matrix_sub(Yp,QjQjtQp);

            // Qp = qr(Yp,0);
            QR_factorization_getQ(Yp, Qp);

            gettimeofday(&end_timeval, NULL);
            elapsed_time = get_seconds_frac(start_timeval,end_timeval);
            //printf("elapsed time for reorthogonalization: %4.8f sec\n", elapsed_time);

            free(QjQjtQp);
            free(QjtQp);
            free(Qj);
            free(inds_global);
        }


        // Bp = Qp'*A
        gettimeofday(&start_timeval, NULL);
        matrix_transpose_matrix_mult(Qp,A,Bp);
        gettimeofday(&end_timeval, NULL);
        elapsed_time = get_seconds_frac(start_timeval,end_timeval);
        //printf("elapsed time for forming Bp: %4.8f sec\n", elapsed_time);


        // A = A - Qp*Bp
        gettimeofday(&start_timeval, NULL);
        matrix_matrix_mult(Qp,Bp,QpBp);
        matrix_sub(A,QpBp); 
        gettimeofday(&end_timeval, NULL);
        elapsed_time = get_seconds_frac(start_timeval,end_timeval);
        //printf("elapsed time for updating A: %4.8f sec\n", elapsed_time);


        // Q(:,ind) = Qp; B(ind,:) = Bp;
        gettimeofday(&start_timeval, NULL);
        matrix_set_selected_columns(*Q, inds_local, Qp);
        matrix_set_selected_rows(*B, inds_local, Bp);
        gettimeofday(&end_timeval, NULL);
        elapsed_time = get_seconds_frac(start_timeval,end_timeval);
        //printf("elapsed time for updating Q and B: %4.8f sec\n", elapsed_time);
        
        free(inds_local);
    }


    // clean up
    matrix_delete(A);
    matrix_delete(RNp); 
    matrix_delete(Yp);
    matrix_delete(Qp);
    matrix_delete(Bp);
    matrix_delete(QpBp);
    matrix_delete(AtQp);
    matrix_delete(AtQp2);
}

