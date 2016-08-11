/* 
driver 5: test various svd routines 
*/

#include "rank_revealing_algorithms_intel_mkl.h"

int main(){
    int i, j, m, n, k, l, p, q, s, kstep, nstep, estep, frank;
    double normM,percent_error,TOL,elapsed_secs;
    mat *M, *C, *U, *R, *Q, *B, *Qk, *Rk, *T, *S, *V;
    vec *I, *Icol, *Irow;
    //time_t start_time, end_time;
    struct timeval start_timeval, end_timeval;


    //  load matrix    
    char *M_file = "../../matrix_data/A_mat_6kx12k.bin"; // matrix filename
    
    printf("loading matrix from %s\n", M_file);
    M = matrix_load_from_binary_file(M_file);
    m = M->nrows;
    n = M->ncols;
    printf("sizes of M are %d by %d\n", m, n);
    printf("norm(M,fro) = %f\n", get_matrix_frobenius_norm(M));

    
    printf("\n %%%%%%%% doing rand low rank SVD full, rand, blockrand.. %%%%%%%% \n");
    q = 2;
    s = 1;
    p = 200;
    for(i=1; i<=1; i++){
        k = i*800;
        gettimeofday(&start_timeval, NULL);
        low_rank_svd_rand_decomp_fixed_rank(M, k, p, 1, q, s, &frank, &U, &S, &V);
        gettimeofday(&end_timeval, NULL);
        elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
        printf("\n===results for low_rank_svd_rand_decomp_fixed_rank1 for k = %d\n",k);
        printf(">>> elapsed time is: %f\n", elapsed_secs);
        use_low_rank_svd_for_approximation(M, U, S, V);
        printf("\n");

        p = 0;
        gettimeofday(&start_timeval, NULL);
        low_rank_svd_rand_decomp_fixed_rank(M, k, p, 2, q, s, &frank, &U, &S, &V);
        gettimeofday(&end_timeval, NULL);
        elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
        printf("\n===results for low_rank_svd_rand_decomp_fixed_rank2 for k = %d\n",k);
        printf(">>> elapsed time is: %f\n", elapsed_secs);
        use_low_rank_svd_for_approximation(M, U, S, V);
        printf("\n");

        
        p = 0;
        kstep = 100;
        gettimeofday(&start_timeval, NULL);
        low_rank_svd_blockrand_decomp_fixed_rank_or_prec(M, k, p, TOL, 2, 
            kstep, q, s, &frank, &U, &S, &V);
        gettimeofday(&end_timeval, NULL);
        elapsed_secs = get_seconds_frac(start_timeval,end_timeval);
        printf("\n===results for low_rank_svd_rand_decomp_fixed_rank3 for k = %d\n",k);
        use_low_rank_svd_for_approximation(M, U, S, V);
    }

    // clean up and exit
    matrix_delete(M); matrix_delete(U); matrix_delete(S); matrix_delete(V);
    return 0;
}

