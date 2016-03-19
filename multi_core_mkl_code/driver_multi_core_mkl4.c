/* 
driver 4: test various tolerance routines 
*/

#include "rank_revealing_algorithms_intel_mkl.h"

int main(){
    int i, j, m, n, k, l, p, q, s, kstep, nstep, estep, frank;
    double normM,percent_error,TOL,elapsed_secs;
    mat *M, *C, *U, *R, *Q, *B, *Qk, *Rk, *T, *S;
    vec *I, *Icol, *Irow;
    //time_t start_time, end_time;
    struct timeval start_timeval, end_timeval;


    // set up vars    
    char *M_file = "../data/A_mat_1kx1k.bin"; // matrix filename
    k = 0; // rank to use : 0 since we use TOL instead
    p = 100; // oversampling
    q = 2; // power for power scheme 
    s = 1; // orthogonalization amount for power scheme 
    kstep = 100; // block size for blockrand methods

    
    printf("loading matrix from %s\n", M_file);
    M = matrix_load_from_binary_file(M_file);
    m = M->nrows;
    n = M->ncols;
    printf("sizes of M are %d by %d\n", m, n);
    printf("norm(M,fro) = %f\n", get_matrix_frobenius_norm(M));


    printf("\n %%%%%%%% doing partial pivoted QR.. %%%%%%%% \n");
    k= 0;
    TOL = .009530; 
    gettimeofday(&start_timeval, NULL);
    pivoted_QR_of_specified_rank_or_prec(M, k, TOL, &frank, &Qk, &Rk, &I);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("\n===results for pivoted_QR_of_specified_rank_or_prec\n");
    printf("output rank is: %d\n", frank);
    printf("elapsed time is: %f\n", elapsed_secs);
    use_pivoted_QR_decomp_for_approximation(M, Qk, Rk, I);


    // test QB factorization
    nstep = 0;
    TOL = 0.03;
    printf("\n %%%%%%%% doing TOL based QB.. %%%%%%%% \n");
    gettimeofday(&start_timeval, NULL);
    randQB_pb_new(M, kstep, nstep, TOL, q, s, &frank, &Q, &B);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("size of Q is: %d x %d\n", Q->nrows, Q->ncols);
    printf("size of B is: %d x %d\n", B->nrows, B->ncols);
    printf("\n===results for randQB_pb_new\n");
    printf("output rank is: %d\n", frank);
    printf("elapsed time is: %f\n", elapsed_secs);
    use_QB_decomp_for_approximation(M, Q, B);


    // test one sided ID  
    k = 0; 
    TOL = .009530; 
    printf("\n %%%%%%%% doing TOL based one sided ID.. %%%%%%%% \n");
    gettimeofday(&start_timeval, NULL);
    id_decomp_fixed_rank_or_prec(M, k, TOL, &frank, &I, &T);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("\n===results for id_decomp_fixed_rank_or_prec\n");
    printf("output rank is: %d\n", frank);
    printf("elapsed time is: %f\n", elapsed_secs);
    use_id_decomp_for_approximation(M, T, I, frank);
    

    printf("\n %%%%%%%% doing TOL based one sided blockrand ID.. %%%%%%%% \n");
    TOL = 0.03;
    gettimeofday(&start_timeval, NULL);
    id_blockrand_decomp_fixed_rank_or_prec(M, k, p, TOL, kstep, q, s, &frank, &I, &T);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("\n===results for id_blockrand_decomp_fixed_rank_or_prec\n");
    printf("output rank is: %d\n", frank);
    printf("elapsed time is: %f\n", elapsed_secs);
    use_id_decomp_for_approximation(M, T, I, frank);
    

    // test two sided ID  
    printf("\n %%%%%%%% doing TOL based two sided ID.. %%%%%%%% \n");
    TOL = .009530; 
    gettimeofday(&start_timeval, NULL);
    id_two_sided_decomp_fixed_rank_or_prec(M, k, TOL, &frank, &Icol, &Irow, &T, &S);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("\n===results for id_two_sided_decomp_fixed_rank_or_prec\n");
    printf("output rank is: %d\n", frank);
    printf("elapsed time is: %f\n", elapsed_secs);
    use_id_two_sided_decomp_for_approximation(M, T, S, Icol, Irow, frank);


    printf("\n %%%%%%%% doing TOL based blockrand two sided ID.. %%%%%%%% \n");
    TOL = 0.03;
    gettimeofday(&start_timeval, NULL);
    id_two_sided_blockrand_decomp_fixed_rank_or_prec(M, k, p, TOL, kstep, q, s, &frank, &Icol, &Irow, &T, &S);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("\n===results for id_two_sided_blockrand_decomp_fixed_rank_or_prec\n");
    printf("output rank is: %d\n", frank);
    printf("elapsed time is: %f\n", elapsed_secs);
    use_id_two_sided_decomp_for_approximation(M, T, S, Icol, Irow, frank);


    // test CUR
    printf("\n %%%%%%%% doing TOL based CUR.. %%%%%%%% \n");
    TOL = .009530; 
    gettimeofday(&start_timeval, NULL);
    cur_decomp_fixed_rank_or_prec(M, k, TOL, &frank, &C, &U, &R);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("\n===results for cur_decomp_fixed_rank_or_prec\n");
    printf("output rank is: %d\n", frank);
    printf("elapsed time is: %f\n", elapsed_secs);
    use_cur_decomp_for_approximation(M, C, U, R);

    printf("\n %%%%%%%% doing TOL based blockrand CUR.. %%%%%%%% \n");
    TOL = 0.03;
    gettimeofday(&start_timeval, NULL);
    cur_blockrand_decomp_fixed_rank_or_prec(M, k, p, TOL, kstep, q, s, &frank, &C, &U, &R);
    gettimeofday(&end_timeval, NULL);
    elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
    printf("\n===results for cur_blockrand_decomp_fixed_rank_or_prec\n");
    printf("output rank is: %d\n", frank);
    printf("elapsed time is: %f\n", elapsed_secs);

    use_cur_decomp_for_approximation(M, C, U, R);


    // clean up and exit
    matrix_delete(Qk); matrix_delete(Rk); matrix_delete(C); matrix_delete(U); matrix_delete(R);
    matrix_delete(M); matrix_delete(T); matrix_delete(S);
    vector_delete(I); vector_delete(Icol); vector_delete(Irow);
    return 0;
}

