% computes low rank SVD of A such that A \approx U*S*V'
% uses blocked QB algorithm to build Q instead of traditional QR
% this can be faster if block size (kstep) is chosen wisely
% k should be approx. kstep*nstep
% q is power scheme sampling parameter i.e. q=2 or 3 is generally sufficient
function [U,S,V] = rsvd_mkl_mex_interface2(A,kstep,nstep,q)
    m = size(A,1); n = size(A,2);
    if kstep*nstep > min(m,n)
        fprintf('kstep*nstep should be <= min(m,n)\n');
        return;
    end

    % warm up
    B = randn(10,10);
    start_mkl_mex(B);         
    pause(0.2);

    % call mex file
    [U,S,V] = rsvd_mkl_mex2(A,kstep,nstep,q);    
end
