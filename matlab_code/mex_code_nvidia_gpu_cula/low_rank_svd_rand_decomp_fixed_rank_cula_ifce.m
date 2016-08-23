% interface to cula based mex file
% computes low rank SVD of A such that A \approx U*S*V'
% k : target rank
% p : oversampling parameter
% vnum : alg to use (1 or 2)
% q : power scheme sampling parameter (q >= 0, integer)
% s: power scheme re-orthonormalization parameter (s >= 1, integer)
function [U,S,V] = low_rank_svd_rand_decomp_fixed_rank_cula_ifce(A,k,p,vnum,q,s)
    m = size(A,1); n = size(A,2);
    if k > min(m,n)
        fprintf('k should be <= min(m,n)\n');
        return;
    end

    % warm up
    %B = randn(10,10);
    %start_mkl_mex(B);         
    %pause(0.2);

    % call mex file
    [U,S,V] = low_rank_svd_rand_decomp_fixed_rank_cula_mex(A,k,p,vnum,q,s);    
end

