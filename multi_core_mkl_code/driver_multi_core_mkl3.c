/* Intel MKL code with OpenMP : 
test driver 3 for interpolative decomposition 
and CUR non-randomized and randomized routines */

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

#include "rank_revealing_algorithms_intel_mkl.h"

int main()
{
    int i, j, m, n, k, l, p, q, s, frank, kstep;
    double TOL,normM,normU,normS,normV,normP,percent_error;
    mat *M, *T, *S, *C, *U, *R;
    vec *Icol, *Irow;
    time_t start_time, end_time;
    //char *M_file = "../data/A_mat_6kx12k.bin";
    //char *M_file = "../data/A_mat_2kx4k.bin";
    char *M_file = "../data/A_mat_1kx2k.bin";
    //char *M_file = "../data/A_mat_10x8.bin";

    printf("loading matrix from %s\n", M_file);
    M = matrix_load_from_binary_file(M_file);
    m = M->nrows;
    n = M->ncols;
    printf("sizes of M are %d by %d\n", m, n);
    printf("norm(M,fro) = %f\n", get_matrix_frobenius_norm(M));

    // now test rank k ID of M..
    k = 400; // rank
    p = 20; // oversampling
    q = 3; // power scheme power
    s = 1; // power scheme orthogonalization amount
    kstep = 100; //block step size
    TOL = 0;
    
    printf("\ncalling rank %d column ID routine..\n", k);
    time(&start_time);
    //id_decomp_fixed_rank(M, k, &Icol, &T);
    id_decomp_fixed_rank_or_prec(M, k, 0, &frank, &Icol, &T);
    time(&end_time);
    printf("elapsed time: about %d seconds\n", (int)difftime(end_time,start_time));
    printf("check error\n");
    use_id_decomp_for_approximation(M, T, Icol, k);


    printf("\ncalling rank %d randomized column ID routine..\n", k);
    time(&start_time);
    p = 20;
    id_rand_decomp_fixed_rank(M, k, p, q, s, &Icol, &T);
    time(&end_time);
    printf("elapsed time: about %d seconds\n", (int)difftime(end_time,start_time));
    printf("check error\n");
    use_id_decomp_for_approximation(M, T, Icol, k);


    printf("\ncalling rank %d block randomized column ID routine..\n", k);
    time(&start_time);
    //id_blockrand_decomp_fixed_rank(M, k, kstep, estep, p, s, &Icol, &T);
    p = 100; // p should be \geq kstep
    id_blockrand_decomp_fixed_rank_or_prec(M, k, p, TOL, kstep, q, s, &frank, &Icol, &T);
    time(&end_time);
    printf("elapsed time: about %d seconds\n", (int)difftime(end_time,start_time));
    printf("check error\n");
    use_id_decomp_for_approximation(M, T, Icol, k);


    printf("\ncalling rank %d two sided ID routine..\n", k);
    time(&start_time);
    //id_two_sided_decomp_fixed_rank(M, k, &Icol, &Irow, &T, &S);
    id_two_sided_decomp_fixed_rank_or_prec(M, k, TOL, &frank, &Icol, &Irow, &T, &S);
    time(&end_time);
    printf("elapsed time: about %d seconds\n", (int)difftime(end_time,start_time));
    printf("check error\n");
    use_id_two_sided_decomp_for_approximation(M, T, S, Icol, Irow, k);


    printf("\ncalling rank %d randomized two sided ID routine..\n", k);
    time(&start_time);
    p = 20;
    id_two_sided_rand_decomp_fixed_rank(M, k, p, q, s, &Icol, &Irow, &T, &S);
    time(&end_time);
    printf("elapsed time: about %d seconds\n", (int)difftime(end_time,start_time));
    printf("check error\n");
    use_id_two_sided_decomp_for_approximation(M, T, S, Icol, Irow, k);

    printf("\ncalling rank %d block randomized two sided ID routine..\n", k);
    time(&start_time);
    p = 100;
    id_two_sided_blockrand_decomp_fixed_rank_or_prec(M, k, p, TOL, kstep, q, s, &frank, &Icol, &Irow, &T, &S);
    time(&end_time);
    printf("elapsed time: about %d seconds\n", (int)difftime(end_time,start_time));
    printf("check error\n");
    use_id_two_sided_decomp_for_approximation(M, T, S, Icol, Irow, k);


    printf("\ncalling rank %d CUR routine\n", k);
    time(&start_time);
    //cur_decomp_fixed_rank(M, k, &C, &U, &R);
    cur_decomp_fixed_rank_or_prec(M, k, TOL, &frank, &C, &U, &R);
    time(&end_time);
    printf("elapsed time: about %d seconds\n", (int)difftime(end_time,start_time));
    printf("check error\n");
    use_cur_decomp_for_approximation(M, C, U, R);

    printf("\ncalling rank %d randomized CUR routine\n", k);
    time(&start_time);
    p = 20;
    cur_rand_decomp_fixed_rank(M, k, p, q, s, &C, &U, &R);
    time(&end_time);
    printf("elapsed time: about %d seconds\n", (int)difftime(end_time,start_time));
    printf("check error\n");
    use_cur_decomp_for_approximation(M, C, U, R);
 
    printf("\ncalling rank %d block randomized CUR routine\n", k);
    time(&start_time);
    p = 100;
    cur_blockrand_decomp_fixed_rank_or_prec(M, k, p, TOL, kstep, q, s, &frank, &C, &U, &R);
    time(&end_time);
    printf("elapsed time: about %d seconds\n", (int)difftime(end_time,start_time));
    printf("check error\n");
    use_cur_decomp_for_approximation(M, C, U, R);
    

    // delete and exit
    printf("delete and exit..\n");
    matrix_delete(M); matrix_delete(T); matrix_delete(S);
    vector_delete(Icol); vector_delete(Irow);

    return 0;
}

