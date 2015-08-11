% computes low rank SVD of A such that A \approx U*S*V'
function [U,S,V] = rsvd_mex_interface1(A,k)
    % warm up
    B = randn(10,10);
    start_mkl_mex(B);         
    pause(2);

    % call mex file
    [U,S,V] = rsvd_mex1(A,k);    
end
