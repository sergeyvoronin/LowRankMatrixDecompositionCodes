/* 
driver 1: use random Gaussian matrices of specified size and run full QR,
partial pivoted QR and single vector and blocked randQB algorithms
over one or more trials and record averaged runtimes
*/

#include "rank_revealing_algorithms_intel_mkl.h"


int main()
{
    int i, j, m, n, k, kstep, frank, tn, ntrials;
    double normM,normU,normS,normV,normP,percent_error;
    mat *M, *Qk, *Rk, *P, *Porig, *QkRk, *QkRkPt, *Q, *B, *QB;
    vec *I, *col_vec;
    //time_t start_time, end_time;
    struct timeval start_timeval, end_timeval;
    double elapsed_secsQR1, elapsed_secsQR2, elapsed_secsQB0, elapsed_secsQB1, elapsed_secsQB2, elapsed_secsQBb0, elapsed_secsQBb1, elapsed_secsQBb2;
    double elapsed_secsQR1_sum, elapsed_secsQR2_sum, elapsed_secsQB0_sum, elapsed_secsQB1_sum, elapsed_secsQB2_sum, elapsed_secsQBb0_sum, elapsed_secsQBb1_sum, elapsed_secsQBb2_sum;
    FILE *fp;


    // set up vars - filename to record timings, number of trials, k and block size
    fp = fopen("timings/driver_multi_core_mkl2.txt","a");
    ntrials = 2;
    k = 100;
    kstep = 20;

    // warm up MKL libs using full pivoted QR ----> 
    printf("warming up using full QR..\n");
    n = 3000;
    m = n;
    printf("making %d x %d matrix..\n", m, n);
    M = matrix_new(m,n);
    initialize_random_matrix(M);
    printf("norm(M,fro) = %f\n", get_matrix_frobenius_norm(M));

    printf("do full QR..\n");
    gettimeofday(&start_timeval, NULL);
    pivotedQR_mkl(M, &Qk, &Rk, &I);
    gettimeofday(&end_timeval, NULL);
    printf("full QR elapsed time: about %4.4f seconds\n", get_seconds_frac(start_timeval,end_timeval));
    matrix_delete(M); matrix_delete(Qk); matrix_delete(Rk); vector_delete(I);
    

    // run all algs for these nxn matrix sizes
    //for(n = 5000; n<=5000; n+=1000){
    for(n = 2000; n<=2000; n+=1000){

        printf("======= n = %d =========\n", n);
        elapsed_secsQR1_sum = 0;
        elapsed_secsQR2_sum = 0;
        elapsed_secsQB0_sum = 0;
        elapsed_secsQB1_sum = 0;
        elapsed_secsQB2_sum = 0;
        elapsed_secsQBb0_sum = 0;
        elapsed_secsQBb1_sum = 0;
        elapsed_secsQBb2_sum = 0;


        // make mats
        m = n;
        QB = matrix_new(m,n);
        QkRk = matrix_new(m,n);
        QkRkPt = matrix_new(m,n);
        Porig = matrix_new(n,n);
        P = matrix_new(n,n);

    
        for(tn = 0; tn<ntrials; tn++){
            
            printf("======= TRIAL %d of %d =========\n", tn+1, ntrials);

            m = n;
            printf("making %d x %d matrix..\n", m, n);
            M = matrix_new(m,n);
            initialize_random_matrix(M);
            printf("norm(M,fro) = %f\n", get_matrix_frobenius_norm(M));


            // full pivoted QR ----> 
            printf("do full QR..\n");
            gettimeofday(&start_timeval, NULL);
            pivotedQR_mkl(M, &Qk, &Rk, &I);
            gettimeofday(&end_timeval, NULL);
            elapsed_secsQR1 = get_seconds_frac(start_timeval,end_timeval);
            printf("full QR elapsed time: about %4.4f seconds\n", get_seconds_frac(start_timeval,end_timeval));


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
            //printf("output rank: %d\n", frank);
            printf("percent_error between M and QkRkPt = %f\n", percent_error);    
            printf("QR elapsed time: about %4.4f seconds\n", get_seconds_frac(start_timeval,end_timeval));
            matrix_delete(Qk); matrix_delete(Rk); vector_delete(I);



            printf("do partial pivoted QR..\n");
            gettimeofday(&start_timeval, NULL);
            pivoted_QR_of_specified_rank(M, k, &frank, &Qk, &Rk, &I);
            gettimeofday(&end_timeval, NULL);
            elapsed_secsQR2 = get_seconds_frac(start_timeval,end_timeval);


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
            //printf("output rank: %d\n", frank);
            printf("percent_error between M and QkRkPt = %f\n", percent_error);    
            printf("QR elapsed time: about %4.4f seconds\n", get_seconds_frac(start_timeval,end_timeval));
            matrix_delete(Qk); matrix_delete(Rk); vector_delete(I);



            // do QB single vector ----->
            printf("doing single vec QB p = 0..\n");
            gettimeofday(&start_timeval, NULL);
            randQB_p(M, k, 0, &Q, &B);
            gettimeofday(&end_timeval, NULL);
            elapsed_secsQB0 = get_seconds_frac(start_timeval,end_timeval);
            matrix_matrix_mult(Q,B,QB);
            percent_error = get_percent_error_between_two_mats(M,QB);
            printf("single vec percent_error between M and QB = %f\n", percent_error);
            printf("single vec elapsed time = %f\n", elapsed_secsQB0);
            matrix_delete(Q); matrix_delete(B);


            printf("doing single vec QB p = 1..\n");
            gettimeofday(&start_timeval, NULL);
            randQB_p(M, k, 1, &Q, &B);
            gettimeofday(&end_timeval, NULL);
            elapsed_secsQB1 = get_seconds_frac(start_timeval,end_timeval);
            matrix_matrix_mult(Q,B,QB);
            percent_error = get_percent_error_between_two_mats(M,QB);
            printf("single vec percent_error between M and QB = %f\n", percent_error);
            printf("single vec elapsed time = %f\n", elapsed_secsQB1);
            matrix_delete(Q); matrix_delete(B);


            printf("doing single vec QB p = 2..\n");
            gettimeofday(&start_timeval, NULL);
            randQB_p(M, k, 2, &Q, &B);
            gettimeofday(&end_timeval, NULL);
            elapsed_secsQB2 = get_seconds_frac(start_timeval,end_timeval);
            matrix_matrix_mult(Q,B,QB);
            percent_error = get_percent_error_between_two_mats(M,QB);
            printf("single vec percent_error between M and QB = %f\n", percent_error);
            printf("single vec elapsed time = %f\n", elapsed_secsQB2);
            matrix_delete(Q); matrix_delete(B);


            
            // do QB blocked ----->
            printf("doing blocked QB p = 0..\n");
            gettimeofday(&start_timeval, NULL);
            randQB_pb(M, kstep, (int)(k/kstep), 0, 1, &Q, &B);
            gettimeofday(&end_timeval, NULL);
            elapsed_secsQBb0 = get_seconds_frac(start_timeval,end_timeval);
            matrix_matrix_mult(Q,B,QB);
            percent_error = get_percent_error_between_two_mats(M,QB);
            printf("blocked percent_error between M and QB = %f\n", percent_error);
            printf("blocked elapsed time = %f\n", elapsed_secsQBb0);
            matrix_delete(Q); matrix_delete(B);


            printf("doing blocked QB p = 1..\n");
            gettimeofday(&start_timeval, NULL);
            randQB_pb(M, kstep, (int)(k/kstep), 1, 1, &Q, &B);
            gettimeofday(&end_timeval, NULL);
            elapsed_secsQBb1 = get_seconds_frac(start_timeval,end_timeval);
            matrix_matrix_mult(Q,B,QB);
            percent_error = get_percent_error_between_two_mats(M,QB);
            printf("blocked percent_error between M and QB = %f\n", percent_error);
            printf("blocked elapsed time = %f\n", elapsed_secsQBb1);
            matrix_delete(Q); matrix_delete(B);


            printf("doing blocked QB p = 2..\n");
            gettimeofday(&start_timeval, NULL);
            randQB_pb(M, kstep, (int)(k/kstep), 2, 1, &Q, &B);
            gettimeofday(&end_timeval, NULL);
            elapsed_secsQBb2 = get_seconds_frac(start_timeval,end_timeval);
            matrix_matrix_mult(Q,B,QB);
            percent_error = get_percent_error_between_two_mats(M,QB);
            printf("blocked percent_error between M and QB = %f\n", percent_error);
            printf("blocked elapsed time = %f\n", elapsed_secsQBb2);
            matrix_delete(Q); matrix_delete(B);


            printf("\n------- n = %d and k = %d --------\n", n,k);
            printf("elapsed time QR1: %4.4f\n", elapsed_secsQR1); 
            printf("elapsed time QR2: %4.4f\n", elapsed_secsQR2); 
            printf("elapsed time QB0: %4.4f\n", elapsed_secsQB0); 
            printf("elapsed time QB1: %4.4f\n", elapsed_secsQB1); 
            printf("elapsed time QB2: %4.4f\n", elapsed_secsQB2); 
            printf("elapsed time QBb0: %4.4f\n", elapsed_secsQBb0); 
            printf("elapsed time QBb1: %4.4f\n", elapsed_secsQBb1); 
            printf("elapsed time QBb2: %4.4f\n", elapsed_secsQBb2); 


            // sum 
            elapsed_secsQR1_sum = elapsed_secsQR1_sum + elapsed_secsQR1;
            elapsed_secsQR2_sum = elapsed_secsQR2_sum + elapsed_secsQR2;
            elapsed_secsQB0_sum = elapsed_secsQB0_sum + elapsed_secsQB0;
            elapsed_secsQB1_sum = elapsed_secsQB1_sum + elapsed_secsQB1;
            elapsed_secsQB2_sum = elapsed_secsQB2_sum + elapsed_secsQB2;
            elapsed_secsQBb0_sum = elapsed_secsQBb0_sum + elapsed_secsQBb0;
            elapsed_secsQBb1_sum = elapsed_secsQBb1_sum + elapsed_secsQBb1;
            elapsed_secsQBb2_sum = elapsed_secsQBb2_sum + elapsed_secsQBb2;


            // delete current matrix
            matrix_delete(M);

        // end of tn loop
        }

        fprintf(fp, "%d  %4.4f   %4.4f   %4.4f   %4.4f  %4.4f  %4.4f  %4.4f  %4.4f\n", n, elapsed_secsQR1_sum/ntrials, elapsed_secsQR2_sum/ntrials, elapsed_secsQB0_sum/ntrials, elapsed_secsQB1_sum/ntrials, elapsed_secsQB2_sum/ntrials, elapsed_secsQBb0_sum/ntrials, elapsed_secsQBb1_sum/ntrials, elapsed_secsQBb2_sum/ntrials);
        fflush(fp);

        // delete working mats and vectors for this n
        matrix_delete(QkRk);
        matrix_delete(QkRkPt);
        matrix_delete(QB);
        matrix_delete(Porig);
        matrix_delete(P);


    // end of n loop
    }

    fclose(fp);

    return 0;
}

