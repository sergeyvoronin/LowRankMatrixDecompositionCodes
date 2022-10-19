/*  
   driver 3: test ID and randomized QB decomp in TOL mode 
*/

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

#include "rank_revealing_algorithms_one_api.h"
#include <stdint.h>
#include <inttypes.h>
#include <omp.h>


int main()
{
    myint64 i, j, k, m, n, kstep, nstep, p, q, s, vnum, offset;
    myint64 frank, numnnz;
    float val,normM,normU,normS,normV,normP,TOL,percent_error;
    mat *M, *D, *MD, *MDMT, *Q, *B, *T;
    vec *svals, *Icol;
    double start_time, end_time;

    m = 5000;
    n = 1000;
    numnnz = m*n; // dense

    M = matrix_new(m,n);
    m = M->nrows;
    n = M->ncols;
    printf("sizes of M are %ld by %ld\n", m, n);
    printf("m*n = %" PRId64 "\n", numnnz); // use 64 bit int 

    printf("initializing random matrix..\n");
    printf("--- calling initialize random matrix ---\n");
    start_time = dsecnd(); // or use omp_get_wtime
    initialize_random_matrix(M);
    end_time = dsecnd();
    printf("--- done with initialization ---\n");
    printf("elapsed time: about %4.2f seconds\n", get_seconds_frac(start_time,end_time));


    printf("compute MDM^T (mirror QDQ^T) --\n");
    //can call QR_factorization_getQ(M,Q);
    start_time = omp_get_wtime();
    D = matrix_new(n,n);
    svals = vector_new(n);
    printf("set svals to decay\n");
    vector_set_element(svals,0,1.0);
    for(i=1; i<n; i++){
        vector_set_element(svals,i, vector_get_element(svals,i-1)/1.02);
    }
    initialize_diagonal_matrix(D, svals);
    MD = matrix_new(m,n);
    MDMT = matrix_new(m,m);
    printf("compute MD\n");
    matrix_matrix_mult(M,D,MD);
    printf("compute MDMT\n");
    matrix_matrix_transpose_mult(MD,M,MDMT);
    matrix_delete(M);
    matrix_delete(MD);
    end_time = omp_get_wtime();;
    printf("elapsed time: about %4.2f seconds\n", end_time - start_time);

    // call ID
    k = 1000; // rank
    p = 20; // oversampling
    q = 1; // power scheme power
    s = 2; // power scheme orthogonalization amount

    printf("computing approximate rank %ld ID..\n", k);
    start_time = omp_get_wtime();
    id_rand_decomp_fixed_rank(MDMT, k, p, q, s, &Icol, &T);
    end_time = omp_get_wtime();
    printf("elapsed time: about %4.2f seconds\n", end_time - start_time);
    printf("normT = %f\n", get_matrix_frobenius_norm(T));

    printf("check ID approximation:\n");
    start_time = omp_get_wtime();
    use_id_decomp_for_approximation(MDMT, T, Icol, k);
    end_time = omp_get_wtime();
    printf("elapsed time: about %4.2f seconds\n", end_time - start_time);

	printf("call QB decomp in TOL mode..\n");
	kstep = 200;
    nstep = -1; // ceil((k+p)/kstep);
	TOL = 1e-2;
    randQB_pb2(MDMT, kstep, nstep, TOL, q, s, &frank, &Q, &B);
	printf("output frank = %ld\n", frank);
	printf("norm(Q) = %f, norm(B) = %f\n", get_matrix_frobenius_norm(Q), get_matrix_frobenius_norm(B));
    use_QB_decomp_for_approximation(MDMT, Q, B);

	printf("use QB outputs to construct ID..\n");
    start_time = omp_get_wtime();
	id_rand_decomp_fromQB(Q, B, &Icol, &T);
    printf("normT = %f\n", get_matrix_frobenius_norm(T));
    printf("normIcol = %f\n", vector_get2norm(Icol));
	printf("check ID approximation:\n");
    use_id_decomp_for_approximation(MDMT, T, Icol, frank);
    end_time = omp_get_wtime();
    printf("elapsed time: about %4.2f seconds\n", end_time - start_time);

    printf("delete and exit..\n");
    matrix_delete(MDMT);
    matrix_delete(Q);
    matrix_delete(B);
    matrix_delete(T);
    vector_delete(Icol);
 
    return 0;
}

