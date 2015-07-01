/* Intel MKL code with OpenMP */

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

#include "rank_revealing_algorithms_nvidia_cula.h"


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

    culaStatus status;

    printf("Initializing CULA\n");
    status = culaInitialize();
    checkStatus(status);



    // set up vars
    fp = fopen("timings/timings_blocked100.txt","a");
    ntrials = 2;
    k = 100;
    kstep = 20;

    for(n = 20000; n<=25000; n+=1000){

        printf("======= n = %d =========\n", n);
        elapsed_secsQBb0_sum = 0;
        elapsed_secsQBb1_sum = 0;
        elapsed_secsQBb2_sum = 0;


        // make mats
        m = n;
        QB = matrix_new(m,n);

    
        for(tn = 0; tn<ntrials; tn++){
            
            printf("======= TRIAL %d of %d =========\n", tn+1, ntrials);

            m = n;
            printf("making %d x %d matrix..\n", m, n);
            M = matrix_new(m,n);
            initialize_random_matrix(M);
            printf("norm(M,fro) = %f\n", get_matrix_frobenius_norm(M));


            // do QB blocked ----->
            printf("doing blocked QB p = 0..\n");
            gettimeofday(&start_timeval, NULL);
            randQB_pb(M, kstep, (int)(k/kstep), 0, &Q, &B);
            gettimeofday(&end_timeval, NULL);
            elapsed_secsQBb0 = get_seconds_frac(start_timeval,end_timeval);
            printf("form QB..\n");
            matrix_matrix_mult(Q,B,QB);
            printf("norm(Q) = %f\n", get_matrix_frobenius_norm(Q));
            printf("norm(B) = %f\n", get_matrix_frobenius_norm(B));
            printf("norm(QB) = %f\n", get_matrix_frobenius_norm(QB));
            printf("norm(M) = %f\n", get_matrix_frobenius_norm(M));
            printf("get percent error..\n");
            percent_error = get_percent_error_between_two_mats(M,QB);
            printf("blocked percent_error between M and QB = %f\n", percent_error);
            printf("blocked elapsed time = %f\n", elapsed_secsQBb0);
            matrix_delete(Q); matrix_delete(B);


            printf("doing blocked QB p = 1..\n");
            gettimeofday(&start_timeval, NULL);
            randQB_pb(M, kstep, (int)(k/kstep), 1, &Q, &B);
            gettimeofday(&end_timeval, NULL);
            elapsed_secsQBb1 = get_seconds_frac(start_timeval,end_timeval);
            matrix_matrix_mult(Q,B,QB);
            percent_error = get_percent_error_between_two_mats(M,QB);
            printf("blocked percent_error between M and QB = %f\n", percent_error);
            printf("blocked elapsed time = %f\n", elapsed_secsQBb1);
            matrix_delete(Q); matrix_delete(B);


            printf("doing blocked QB p = 2..\n");
            gettimeofday(&start_timeval, NULL);
            randQB_pb(M, kstep, (int)(k/kstep), 2, &Q, &B);
            gettimeofday(&end_timeval, NULL);
            elapsed_secsQBb2 = get_seconds_frac(start_timeval,end_timeval);
            matrix_matrix_mult(Q,B,QB);
            percent_error = get_percent_error_between_two_mats(M,QB);
            printf("blocked percent_error between M and QB = %f\n", percent_error);
            printf("blocked elapsed time = %f\n", elapsed_secsQBb2);
            matrix_delete(Q); matrix_delete(B);


            printf("\n------- n = %d and k = %d --------\n", n,k);
            printf("elapsed time QBb0: %4.4f\n", elapsed_secsQBb0); 
            printf("elapsed time QBb1: %4.4f\n", elapsed_secsQBb1); 
            printf("elapsed time QBb2: %4.4f\n", elapsed_secsQBb2); 


            // sum 
            elapsed_secsQBb0_sum = elapsed_secsQBb0_sum + elapsed_secsQBb0;
            elapsed_secsQBb1_sum = elapsed_secsQBb1_sum + elapsed_secsQBb1;
            elapsed_secsQBb2_sum = elapsed_secsQBb2_sum + elapsed_secsQBb2;


            // delete current matrix
            matrix_delete(M);

        // end of tn loop
        }

        fprintf(fp, "%d  %4.4f   %4.4f   %4.4f\n", n, elapsed_secsQBb0_sum/ntrials, elapsed_secsQBb1_sum/ntrials, elapsed_secsQBb2_sum/ntrials);
        fflush(fp);

        // delete working mats and vectors for this n
        matrix_delete(QB);


    // end of n loop
    }

    fclose(fp);


    printf("Shutting down CULA\n");
    culaShutdown();

    return EXIT_SUCCESS;
}

