#include "rank_revealing_algorithms_intel_mkl.h"


/* computes the approximate low rank SVD of rank k of matrix M using BBt version */
void randomized_low_rank_svd1(mat *M, int k, mat **U, mat **S, mat **V){
    int i,j,m,n;
    double val;
    m = M->nrows; n = M->ncols;

    printf("running randomized_low_rank_svd1 with k = %d\n", k);

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
    matrix_delete(BBt);
    matrix_delete(Sinv);
    matrix_delete(UhatSinv);
}



/* computes the approximate low rank SVD of rank k of matrix M using QR version */
void randomized_low_rank_svd2(mat *M, int k, mat **U, mat **S, mat **V){
    int i,j,m,n;
    double val;
    m = M->nrows; n = M->ncols;

    printf("running randomized_low_rank_svd2 with k = %d\n", k);

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
void randomized_low_rank_svd3(mat *M, int k, int q, int s, mat **U, mat **S, mat **V){
    int i,j,m,n;
    double val;
    m = M->nrows; n = M->ncols;

    printf("running randomized_low_rank_svd3 with k = %d, q = %d, s = %d\n", k,q,s);

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
    vector_delete(evals);
    vector_delete(singvals);
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
    //printf("norm(Q,fro) = %f\n", get_matrix_frobenius_norm(Q));

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
    //estimate_rank_and_buildQ(M,frac_of_max_rank,TOL,&Q,&k);
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


/* column pivoted QR for mxn A 
input: matrix M
outputs: matrices Q and R and vector I such that A(:,I) = QR */
void pivotedQR_mkl(mat *M, mat **Q, mat **R, vec **I){
    int i,j,k,m,n,Rrows,Rcols,Qrows,Qcols;
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
    LAPACKE_dgeqp3(CblasColMajor, Mwork->nrows, Mwork->ncols, Mwork->d, Mwork->nrows, Iarr, tau_arr);

    *R = matrix_new(Rrows,Rcols);
    for(i=0; i<Rrows; i++){
        for(j=i; j<Rcols; j++){
            matrix_set_element(*R,i,j,matrix_get_element(Mwork,i,j));
        }
    }

    // get Q
    LAPACKE_dorgqr(CblasColMajor, Mwork->nrows, Mwork->nrows, min(Mwork->nrows,Mwork->ncols), Mwork->d, Mwork->nrows, tau_arr);

    *Q = matrix_new(Qrows,Qcols);
    for(i=0; i<Qrows; i++){
        for(j=0; j<Qcols; j++){
            matrix_set_element(*Q,i,j,matrix_get_element(Mwork,i,j));
        }
    }

    // get I
    *I = vector_new(n);
    for(i=0; i<n; i++){
        vector_set_element(*I,i,Iarr[i]-1);
    }


    // free temp variables
    matrix_delete(Mwork);
    free(Iarr);
    free(tau_arr);
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


/* solve A X = B where A is upper triangular matrix and X is a matrix 
invert different ways
1. using tridiagonal matrix system solve
2. using inverse of tridiagonal matrix solve
3. Use SVD of A to compute inverse 
default: solve column by column with tridiagonal system
*/
/*void upper_triangular_system_solve(mat *A, mat *B, mat *X, int solve_type){
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
        singular_value_decomposition(A, &U, &S, &Vt);
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
}*/


/* computes the column ID decomposition of a matrix of specified rank 
: [I,T] = id_decomp_fixed_rank(M,k) 
where I is the vector from the permutation and T = inv(Rk1)*Rk2 */
void id_decomp_fixed_rank(mat *M, int k, vec **I, mat **T){
    int i,j,frankQR,ind,m,n;
    mat *Qk, *Rk, *Rk1, *Rk2;
    m = M->nrows;
    n = M->ncols;

    /* 
        [Qk,Rk,P,I] = qr_with_column_pivoting_fixed_rank(A,k);
        Rk1 = Rk(:,1:k);
        Rk2 = Rk(:,(k+1):end);
    */
    if( k < min(m,n) ){
        pivoted_QR_of_specified_rank(M, k, &frankQR, &Qk, &Rk, I);
    }else{
        frankQR = k;
        //pivoted_QR_of_specified_rank(M, k, &frankQR, &Qk, &Rk, I);
        pivotedQR_mkl(M, &Qk, &Rk, I);
    }

    Rk1 = matrix_new(frankQR,frankQR);
    Rk2 = matrix_new(frankQR,n-frankQR);
    fill_matrix_from_first_columns(Rk, frankQR, Rk1);
    fill_matrix_from_last_columns(Rk, frankQR, Rk2);

    *T = matrix_new(Rk2->nrows,Rk2->ncols);

    // NOTE: must enforce Rk1 to be upper triangular before inverting..
    // %Rk1*T = Rk2
    matrix_keep_only_upper_triangular(Rk1);
    upper_triangular_system_solve(Rk1,Rk2,*T,1);

    matrix_delete(Rk1);
    matrix_delete(Rk2);
}


/* computes the two sided ID decomposition of a matrix of specified rank 
: [Icol,Irow,T,S] = id_two_sided_decomp_fixed_rank(M,k) 
where Icol is the column redindexing vector and Irow is the row 
indexing vector and T,S the matrices corresponding to column and row IDs */
void id_two_sided_decomp_fixed_rank(mat *M, int k, vec **Icol, vec **Irow, mat **T, mat **S){
    int m,n;
    mat *MI, *MIt;
    m = M->nrows;
    n = M->ncols;

    // perform column ID
    id_decomp_fixed_rank(M, k, Icol, T);

    // form MI
    MI = matrix_new(M->nrows,k);
    MIt = matrix_new(k,M->nrows);
    fill_matrix_from_first_columns_from_list(M, *Icol, k, MI);
    matrix_build_transpose(MIt, MI);

    // perform row ID 
    id_decomp_fixed_rank(MIt, k, Irow, S);

    matrix_delete(MI);
    matrix_delete(MIt);
}


/* computes a rank k cur decomposition of a matrix */
void cur_decomp_fixed_rank(mat *M, int k, mat **C, mat **U, mat **R){
    mat *Ik, *T, *S, *Tt, *V1, *V, *RV, *RRt, *Ut;
    vec *Icol, *Irow, *Icolinv;
    int i,minindex, maxindex;
    double minval, maxval;
    
    // perform two sided ID
    id_two_sided_decomp_fixed_rank(M, k, &Icol, &Irow, &T, &S);

    Ik = matrix_new(k,k);
    initialize_identity_matrix(Ik);

    // build Icolinv
    printf("build Icolinv\n");
    Icolinv = vector_new(Icol->nrows);
    vector_build_rewrapped(Icolinv,Icol);
    
    printf("build Tt\n");
    Tt = matrix_new(T->ncols,T->nrows);
    matrix_build_transpose(Tt,T);

    printf("build V1\n");
    V1 = matrix_new(Ik->nrows + Tt->nrows,Ik->ncols);
    append_matrices_vertically(Ik,Tt,V1);

    // V = V1(Icolinv,:);
    printf("build V\n");
    V = matrix_new(Icolinv->nrows,V1->ncols);
    fill_matrix_from_first_rows_from_list(V1, Icolinv, Icolinv->nrows, V);

    //R = M(Irow(1:k),:)
    printf("build R\n");
    printf("norm(Icol) = %f\n", vector_get2norm(Icol));
    printf("norm(Irow) = %f\n", vector_get2norm(Irow));
    //vector_print(Irow);
    vector_get_min_element(Irow, &minindex, &minval);
    vector_get_max_element(Irow, &maxindex, &maxval);
    printf("minval = %fat i=%d and maxval = %f at i=%d\n", minval, minindex, maxval, maxindex);

    *R = matrix_new(k,M->ncols);
    fill_matrix_from_first_rows_from_list(M, Irow, k, *R);
    //printf("norm(R) = %f\n", get_matrix_frobenius_norm(*R));

    //C = M(:,Icol(1:k));
    printf("build C\n");
    *C = matrix_new(M->nrows,k);
    fill_matrix_from_first_columns_from_list(M, Icol, k, *C);
    //printf("norm(C) = %f\n", get_matrix_frobenius_norm(*C));


    //RRt*Ut = V
    printf("build U\n");
    RRt = matrix_new(k,k);
    Ut = matrix_new(k,k);
    *U = matrix_new(k,k);
    RV = matrix_new(k,k);
    matrix_matrix_transpose_mult(*R,*R,RRt);
    matrix_matrix_mult(*R,V,RV);
    printf("solve for Ut\n");
    square_matrix_system_solve(RRt,Ut,RV);
    printf("transpose to get U\n");
    matrix_build_transpose(*U,Ut);

    matrix_delete(Ut); matrix_delete(RV); matrix_delete(RRt);
    matrix_delete(V1); matrix_delete(V); matrix_delete(T);
    matrix_delete(Tt); matrix_delete(S);
    vector_delete(Irow); vector_delete(Icol); vector_delete(Icolinv);
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


/* evaluate approximation to M using supplied column ID of rank k */
void use_id_decomp_for_approximation(mat *M, mat *T, vec *I, int k){
    vec *Iinv;
    mat *Tt, *Ik, *V1, *V, *MI, *MA;
    int m,n;
    
    m = M->nrows; n = M->ncols;

    printf("norm(T,fro) = %f\n", get_matrix_frobenius_norm(T));
    printf("norm(T,max) = %f\n", get_matrix_max_abs_element(T));

    // build Iinv
    printf("build Iinv\n");
    Iinv = vector_new(I->nrows);
    vector_build_rewrapped(Iinv,I);
    
    printf("build Tt\n");
    Tt = matrix_new(T->ncols,T->nrows);
    matrix_build_transpose(Tt,T);
    //printf("norm(Tt,fro) = %f\n", get_matrix_frobenius_norm(Tt));

    Ik = matrix_new(k,k);
    initialize_identity_matrix(Ik);
    //printf("norm(Ik,fro) = %f\n", get_matrix_frobenius_norm(Ik));

    printf("build V1\n");
    V1 = matrix_new(Ik->nrows + Tt->nrows,Ik->ncols);
    append_matrices_vertically(Ik,Tt,V1);

    // V = V1(Iinv,:);
    printf("build V\n");
    V = matrix_new(Iinv->nrows,V1->ncols);
    fill_matrix_from_first_rows_from_list(V1, Iinv, Iinv->nrows, V);

    // MI = M(:,I(1:k)); 
    printf("build MI\n");
    MI = matrix_new(M->nrows,k);
    fill_matrix_from_first_columns_from_list(M, I, k, MI);

    printf("build MA\n");
    MA = matrix_new(m,n);
    matrix_matrix_transpose_mult(MI,V,MA);

    printf("norm(M,fro) = %f\n", get_matrix_frobenius_norm(M));
    printf("norm(MA,fro) = %f\n", get_matrix_frobenius_norm(MA));
    printf("percent error = %f\n", get_percent_error_between_two_mats(M,MA));

    matrix_delete(Ik); matrix_delete(V1); matrix_delete(V);
    matrix_delete(MI); matrix_delete(MA); matrix_delete(Tt);
    vector_delete(Iinv);
}


/* evaluate approximation to M using supplied two sided ID of rank k */
void use_id_two_sided_decomp_for_approximation(mat *M, mat *T, mat *S, vec *Icol, vec *Irow, int k){
    vec *Icolinv, *Irowinv;
    mat *Tt, *St, *Ik, *V1, *V, *U1, *U, *MI, *MIJ, *UMIJ, *MA;
    int m,n;
    
    m = M->nrows; n = M->ncols;

    printf("norm(T,fro) = %f\n", get_matrix_frobenius_norm(T));
    printf("norm(T,max) = %f\n", get_matrix_max_abs_element(T));
    printf("norm(S,fro) = %f\n", get_matrix_frobenius_norm(S));
    printf("norm(S,max) = %f\n", get_matrix_max_abs_element(S));

    // build kxk identity
    Ik = matrix_new(k,k);
    initialize_identity_matrix(Ik);

    // build Icolinv
    printf("build Icolinv\n");
    Icolinv = vector_new(Icol->nrows);
    vector_build_rewrapped(Icolinv,Icol);
    
    printf("build Tt\n");
    Tt = matrix_new(T->ncols,T->nrows);
    matrix_build_transpose(Tt,T);
    //printf("norm(Tt,fro) = %f\n", get_matrix_frobenius_norm(Tt));

    printf("build V1\n");
    V1 = matrix_new(Ik->nrows + Tt->nrows,Ik->ncols);
    append_matrices_vertically(Ik,Tt,V1);

    // V = V1(Icolinv,:);
    printf("build V\n");
    V = matrix_new(Icolinv->nrows,V1->ncols);
    fill_matrix_from_first_rows_from_list(V1, Icolinv, Icolinv->nrows, V);

    // build Irowinv
    printf("build Irowinv\n");
    Irowinv = vector_new(Irow->nrows);
    vector_build_rewrapped(Irowinv,Irow);
    
    printf("build St\n");
    St = matrix_new(S->ncols,S->nrows);
    matrix_build_transpose(St,S);

    printf("build U1\n");
    U1 = matrix_new(Ik->nrows + St->nrows,Ik->ncols);
    append_matrices_vertically(Ik,St,U1);

    // U = U1(Irowinv,:);
    printf("build U\n");
    U = matrix_new(Irowinv->nrows,V1->ncols);
    fill_matrix_from_first_rows_from_list(U1, Irowinv, Irowinv->nrows, U);

    // MI = M(:,Icol(1:k)); 
    // MIJ = M(Irow(1:k),Icol(1:k)); 
    printf("build MI\n");
    MI = matrix_new(M->nrows,k);
    fill_matrix_from_first_columns_from_list(M, Icol, k, MI);
    MIJ = matrix_new(k,k);
    fill_matrix_from_first_rows_from_list(MI, Irow, k, MIJ);

    printf("build MA\n");
    UMIJ = matrix_new(m,k);
    MA = matrix_new(m,n);
    matrix_matrix_mult(U,MIJ,UMIJ);
    matrix_matrix_transpose_mult(UMIJ,V,MA);

    printf("norm(M,fro) = %f\n", get_matrix_frobenius_norm(M));
    printf("norm(MA,fro) = %f\n", get_matrix_frobenius_norm(MA));
    printf("percent error = %f\n", get_percent_error_between_two_mats(M,MA));

    matrix_delete(Ik); matrix_delete(MI); matrix_delete(MIJ);
    matrix_delete(UMIJ); matrix_delete(MA); matrix_delete(Tt);
    matrix_delete(V1); matrix_delete(V); matrix_delete(U1);
    matrix_delete(U); matrix_delete(St);
    vector_delete(Icolinv);
    vector_delete(Irowinv);
}


/* evaluate approximation to M using supplied CUR decomposition of rank k */
void use_cur_decomp_for_approximation(mat *M, mat *C, mat *U, mat *R){
    mat *P;
    P = matrix_new(M->nrows, M->ncols);
    form_cur_product_matrix(C, U, R, P);

    printf("norm(M,fro) = %f\n", get_matrix_frobenius_norm(M));
    printf("norm(P,fro) = %f\n", get_matrix_frobenius_norm(P));
    printf("percent error = %f\n", get_percent_error_between_two_mats(M,P));

    matrix_delete(P);
}

