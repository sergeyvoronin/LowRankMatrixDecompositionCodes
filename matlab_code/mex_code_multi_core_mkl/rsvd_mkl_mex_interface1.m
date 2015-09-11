% computes low rank SVD of A such that A \approx U*S*V'
function [U,S,V] = rsvd_mkl_mex_interface1(A,k)
    m = size(A,1); n = size(A,2);
    if k > min(m,n)
        fprintf('k should be <= min(m,n)\n');
        return;
    end

    % warm up
    B = randn(10,10);
    start_mkl_mex(B);         
    pause(0.2);

    % call mex file
    [U,S,V] = rsvd_mkl_mex1(A,k);    
end
