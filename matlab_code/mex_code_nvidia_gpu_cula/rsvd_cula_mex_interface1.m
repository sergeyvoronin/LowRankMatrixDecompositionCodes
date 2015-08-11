% computes low rank SVD of A such that A \approx U*S*V'
function [U,S,V] = rsvd_cula_mex_interface1(A,k)
    % call mex file
    [U,S,V] = rsvd_cula_mex1(A,k);    
end
