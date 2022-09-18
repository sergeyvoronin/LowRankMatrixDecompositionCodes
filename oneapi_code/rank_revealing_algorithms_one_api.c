/* matrix decomposition algorithms with Intel oneAPI */
/* Sergey Voronin */

#include "rank_revealing_algorithms_one_api.h"

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
        printf("M M^T mult j=%ld of %ld\n", j, q-1);

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
        printf("resize mats to rank %ld from end\n",k);
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


/* use Q,B outputs to construct low rank SVD */
void low_rank_svd_rand_decomp_fromQB(mat **Q, mat **B, mat **U, mat **S, mat **V){
    myint64 i,m,n,knew; 
    mat *BBt, *Uhat, *St, *Vt, *Sinv, *UhatSinv;
    vec *singvals;
    m = (*Q)->nrows; n = (*B)->ncols;
    knew =(*Q)->ncols;
    //printf("m = %ld, n = %ld, knew = %ld\n", m, n, knew);

    // setup mats
    *U = matrix_new(m,knew);
    *S = matrix_new(knew,knew);
    *V = matrix_new(n,knew);
    Uhat = matrix_new(knew,knew);
    St = matrix_new(knew,knew);
    Vt = matrix_new(knew,n);

    BBt = matrix_new(knew,knew);
    matrix_matrix_transpose_mult(*B,*B,BBt); 

    // compute eigendecomposition of BBt
    //printf("norm(BBt) = %f\n", get_matrix_frobenius_norm(BBt));

    // SVD on small symmetric matrix (replace with eigendecomp)
    singular_value_decomposition(BBt, Uhat, St, Vt);
    /*printf("norm(BBt) = %f\n", get_matrix_frobenius_norm(BBt));
    printf("norm(Uhat) = %f\n", get_matrix_frobenius_norm(Uhat));
    printf("norm(St) = %f\n", get_matrix_frobenius_norm(St));
    printf("norm(Vt) = %f\n", get_matrix_frobenius_norm(Vt));*/

    //printf("form S..\n");
    singvals = vector_new(knew);
    for(i=0; i<knew; i++){
        //printf("sing[%ld] = %f\n", i, sqrt(matrix_get_element(St,i,i)));
        vector_set_element(singvals,i,sqrt(matrix_get_element(St,i,i)));
    }
    initialize_diagonal_matrix(*S, singvals);
    printf("norm(S) = %f\n", get_matrix_frobenius_norm(*S));

    // compute U = Q*Uhat mxk * kxk (mxk)  
    //printf("form U..\n");
    matrix_matrix_mult(*Q,Uhat,*U);
    //printf("norm(U) = %f\n", get_matrix_frobenius_norm(*U));

    // compute V = B^T Uhat * Sigma^{-1} (nxk)
    //printf("form V..\n");
    Sinv = matrix_new(knew,knew);
    UhatSinv = matrix_new(knew,knew);
    invert_diagonal_matrix(Sinv,*S);
    matrix_matrix_mult(Uhat,Sinv,UhatSinv);
    matrix_transpose_matrix_mult(*B,UhatSinv,*V);
    //printf("norm(V) = %f\n", get_matrix_frobenius_norm(*V));

    // clean up
    printf("clean up..\n");
    matrix_delete(BBt);
    matrix_delete(Vt);
    //matrix_delete(*Q);
    //matrix_delete(*B);
    matrix_delete(Uhat);
    matrix_delete(Sinv);
    matrix_delete(UhatSinv);
    vector_delete(singvals);
}


/* randQB blocked algorithm with power method 
inputs: matrix M, integer kstep (block size), integer nstep (number of blocks), power scheme parameter p [ integer (>=0) ], orthogonalization amount in power scheme parameter s
outputs: matrices Q and B s.t. M \approx Q*B 
*/
void randQB_pb(mat *M, myint64 kstep, myint64 nstep, myint64 p, myint64 s, mat **Q, mat **B){
    myint64 i,j,m,n,l,step;
    double dotp,elapsed_time;
    myint64 *inds_local, *inds_global;
    mat *A, *RN, *RNp, *Yp, *Qp, *Bp, *AtQp, *AtQp2, *QpBp, *Qj, *QjtQp, *QjQjtQp;
    vec *ej,*rj,*pj,*qj,*qi,*yj,*bj;
    double start_t, end_t;

    m = M->nrows; n = M->ncols;
    // check to make sure kstep is not too large
    if(kstep > (int64_t) (min(m,n)/2)){
        kstep = (int64_t)min(m,n)/10;
        printf("kstep resized to %ld\n", kstep);
    }
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

    for(step=0; step<nstep; step++){
        printf("in block step %ld\n", step);
        
        start_t = omp_get_wtime(); 
        inds_local = (int64_t*)malloc(kstep*sizeof(int64_t));
        for(i=0; i<kstep; i++){
            inds_local[i] = kstep*step + i;
        } 
        matrix_get_selected_columns(RN, inds_local, RNp);
        matrix_matrix_mult(A,RNp,Yp);
        end_t = omp_get_wtime(); 
        elapsed_time = end_t - start_t;
        //printf("elapsed time for building Yp: %4.8f sec\n", elapsed_time);


        // power method
        start_t = omp_get_wtime(); 
        for(j=1; j<=p; j++){
            if((2*j-2) % s == 0){
                QR_factorization_getQ(Yp, Qp);
                matrix_transpose_matrix_mult(A,Qp,AtQp);
            }
            else{
                matrix_transpose_matrix_mult(A,Yp,AtQp);
            }

            if((2*j-1) % s == 0){
                QR_factorization_getQ(AtQp, AtQp2);
                matrix_matrix_mult(A,AtQp2,Yp);
            }
            else{
                matrix_matrix_mult(A,AtQp,Yp);
            }
        }

        // Qp = qr(Yp,0) 
        QR_factorization_getQ(Yp, Qp);
        end_t = omp_get_wtime(); 
        elapsed_time = end_t - start_t;
        //printf("elapsed time power method: %4.8f sec\n", elapsed_time);


        // project Qp away from previous Q stuff
        if(step>0){
            start_t = omp_get_wtime(); 
            inds_global = (int64_t*)malloc(step*kstep*sizeof(int64_t));
            Qj = matrix_new(m,step*kstep);
            QjtQp = matrix_new(step*kstep,kstep);
            QjQjtQp = matrix_new(m,kstep);

            for(i=0; i<(step*kstep); i++){
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

            end_t = omp_get_wtime(); 
            elapsed_time = end_t - start_t;
            //printf("elapsed time for reorthogonalization: %4.8f sec\n", elapsed_time);

            free(QjQjtQp);
            free(QjtQp);
            free(Qj);
            free(inds_global);
        }

        // Bp = Qp'*A
        start_t = omp_get_wtime(); 
        matrix_transpose_matrix_mult(Qp,A,Bp);
        end_t = omp_get_wtime(); 
        elapsed_time = end_t - start_t;
        //printf("elapsed time for forming Bp: %4.8f sec\n", elapsed_time);


        // A = A - Qp*Bp
        start_t = omp_get_wtime(); 
        matrix_matrix_mult(Qp,Bp,QpBp);
        matrix_sub(A,QpBp); 
        end_t = omp_get_wtime(); 
        elapsed_time = end_t - start_t;
        //printf("elapsed time for updating A: %4.8f sec\n", elapsed_time);

        // Q(:,ind) = Qp; B(ind,:) = Bp;
        start_t = omp_get_wtime(); 
        matrix_set_selected_columns(*Q, inds_local, Qp);
        matrix_set_selected_rows(*B, inds_local, Bp);
        end_t = omp_get_wtime(); 
        elapsed_time = end_t - start_t;
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


/* evaluate approximation to M using supplied QB decomposition */
void use_QB_decomp_for_approximation(mat *M, mat *Q, mat *B){
    mat *P;
    P = matrix_new(M->nrows, M->ncols);
    matrix_matrix_mult(Q, B, P);

    printf("norm(M,fro) = %f\n", get_matrix_frobenius_norm(M));
    printf("norm(P,fro) = %f\n", get_matrix_frobenius_norm(P));
    printf("percent error = %f\n", get_percent_error_between_two_mats(M,P));

    matrix_delete(P);
}

