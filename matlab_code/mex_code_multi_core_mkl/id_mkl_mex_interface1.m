% computes ID of rank k of A such that A(:,I) \approx A(:,I(1:k) [I, T*]*
function [I,T] = id_mkl_mex_interface1(A,k)
    % warm up
    B = randn(10,10);
    start_mkl_mex(B);         
    pause(0.2);

    % call mex file
    [I,T] = id_mkl_mex1(A,k);    
end
