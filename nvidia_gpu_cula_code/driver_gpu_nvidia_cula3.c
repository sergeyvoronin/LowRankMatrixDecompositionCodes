/* NVIDIA CULA code interfaced with Intel MKL code with OpenMP  
test driver 3 for interpolative decomposition 
and CUR non-randomized and randomized routines */

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

#include "rank_revealing_algorithms_nvidia_cula.h"

int main()
{
    int i, j, m, n, k, l, p, q, s, frank, kstep, nstep, culaVersion;
    double TOL,normM,normU,normS,normV,normP,percent_error,elapsed_secs;
    mat *M, *T, *S, *C, *U, *R, *Q, *B;
    vec *Icol, *Irow;
    time_t start_time, end_time;
    char *M_file = "../../matrix_data/A_mat_6kx12k.bin";
    //char *M_file = "../data/A_mat_2kx4k.bin";
    //char *M_file = "../data/A_mat_1kx2k.bin";
    //char *M_file = "../data/A_mat_10x8.bin";
    struct timeval start_timeval, end_timeval;
    culaStatus status;

    printf("Initializing CULA\n");
    status = culaInitialize();
    checkStatus(status);

    culaVersion = culaGetVersion();
    printf("culaVersion is %d\n", culaVersion);
 

    printf("loading matrix from %s\n", M_file);
    M = matrix_load_from_binary_file(M_file);
    m = M->nrows;
    n = M->ncols;
    printf("sizes of M are %d by %d\n", m, n);
    printf("norm(M,fro) = %f\n", get_matrix_frobenius_norm(M));

    // now test rank k ID of M..
    k = 1000; // rank
    p = 20; // oversampling
    q = 0; // power scheme power
    s = 2; // power scheme orthogonalization amount
    kstep = 500; //block step size
    nstep = 2;
    TOL = 0;
    
    printf("\ncalling rank %d column ID routine..\n", k);
    gettimeofday(&start_timeval, NULL);
    //id_decomp_fixed_rank_or_prec(M, k, 0, &frank, &Icol, &T);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("elapsed time is: %4.1f\n", elapsed_secs);
    printf("check error\n");
    //use_id_decomp_for_approximation(M, T, Icol, k);


    printf("\ncalling rank %d old QB routine..\n", k);
    gettimeofday(&start_timeval, NULL);
    //randQB_pb(M, kstep, nstep, q, s, &Q, &B);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("elapsed time is: %4.1f\n", elapsed_secs);
    printf("check error\n");
    //use_QB_decomp_for_approximation(M, Q, B);


    printf("\ncalling rank %d new QB routine..\n", k);
    gettimeofday(&start_timeval, NULL);
    randQB_pb_new(M, kstep, nstep, TOL, q, s, &frank, &Q, &B);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("elapsed time is: %4.1f\n", elapsed_secs);
    printf("check error\n");
    use_QB_decomp_for_approximation(M, Q, B);



    printf("\ncalling rank %d block randomized column ID routine..\n", k);
    p = kstep; // p should be \geq kstep
    gettimeofday(&start_timeval, NULL);
    id_blockrand_decomp_fixed_rank_or_prec(M, k, p, TOL, kstep, q, s, &frank, &Icol, &T);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("elapsed time is: %4.1f\n", elapsed_secs);
    printf("check error\n");
    use_id_decomp_for_approximation(M, T, Icol, k);



    printf("\ncalling rank %d randomized column ID routine..\n", k);
    p = kstep;
    gettimeofday(&start_timeval, NULL);
    id_rand_decomp_fixed_rank(M, k, p, q, s, &Icol, &T);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("elapsed time is: %4.1f\n", elapsed_secs);
    printf("check error\n");
    use_id_decomp_for_approximation(M, T, Icol, k);


    // delete and exit
    //printf("delete and exit..\n");
    //matrix_delete(M); matrix_delete(T); matrix_delete(S);
    //vector_delete(Icol); vector_delete(Irow);

    //printf("Shutting down CULA\n");
    //culaShutdown();
    //return EXIT_SUCCESS;


    printf("\ncalling rank %d two sided ID routine..\n", k);
    gettimeofday(&start_timeval, NULL);
    //id_two_sided_decomp_fixed_rank_or_prec(M, k, TOL, &frank, &Icol, &Irow, &T, &S);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("elapsed time is: %4.1f\n", elapsed_secs);
    printf("check error\n");
    //use_id_two_sided_decomp_for_approximation(M, T, S, Icol, Irow, k);

    printf("\ncalling rank %d randomized two sided ID routine..\n", k);
    p = 20;
    gettimeofday(&start_timeval, NULL);
    id_two_sided_rand_decomp_fixed_rank(M, k, p, q, s, &Icol, &Irow, &T, &S);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("elapsed time is: %4.1f\n", elapsed_secs);
    printf("check error\n");
    use_id_two_sided_decomp_for_approximation(M, T, S, Icol, Irow, k);

    printf("\ncalling rank %d block randomized two sided ID routine..\n", k);
    p = kstep;
    gettimeofday(&start_timeval, NULL);
    id_two_sided_blockrand_decomp_fixed_rank_or_prec(M, k, p, TOL, kstep, q, s, &frank, &Icol, &Irow, &T, &S);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("elapsed time is: %4.1f\n", elapsed_secs);
    printf("check error\n");
    use_id_two_sided_decomp_for_approximation(M, T, S, Icol, Irow, k);


    printf("\ncalling rank %d CUR routine\n", k);
    gettimeofday(&start_timeval, NULL);
    //cur_decomp_fixed_rank_or_prec(M, k, TOL, &frank, &C, &U, &R);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("elapsed time is: %4.1f\n", elapsed_secs);
    printf("check error\n");
    //use_cur_decomp_for_approximation(M, C, U, R);

    printf("\ncalling rank %d randomized CUR routine\n", k);
    p = 20;
    gettimeofday(&start_timeval, NULL);
    cur_rand_decomp_fixed_rank(M, k, p, q, s, &C, &U, &R);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("elapsed time is: %4.1f\n", elapsed_secs);
    printf("check error\n");
    use_cur_decomp_for_approximation(M, C, U, R);
 
    printf("\ncalling rank %d block randomized CUR routine\n", k);
    p = kstep;
    gettimeofday(&start_timeval, NULL);
    cur_blockrand_decomp_fixed_rank_or_prec(M, k, p, TOL, kstep, q, s, &frank, &C, &U, &R);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("elapsed time is: %4.1f\n", elapsed_secs);
    printf("check error\n");
    use_cur_decomp_for_approximation(M, C, U, R);
    

    // delete and exit
    printf("delete and exit..\n");
    matrix_delete(M); matrix_delete(T); matrix_delete(S);
    vector_delete(Icol); vector_delete(Irow);

    printf("Shutting down CULA\n");
    culaShutdown();

    return EXIT_SUCCESS;
}

