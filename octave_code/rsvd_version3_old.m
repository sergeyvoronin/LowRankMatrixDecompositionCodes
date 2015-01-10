% QR of B^T version with sampling of (A A^T)^q A R matrix 
% old version
function [U,Sigma,V] = rsvd_version3_old(A,k,q)
    m = size(A,1);
    n = size(A,2);

    R = randn(n,k);
    Y = A*R; % m \times n * n \times k = m \times k

    Z = Y;
    Q = orth(Z);
    for j=1:q
        Y = A'*Q;
        if mod(j,2) == 0
            W = orth(Y);
            Z = A*W;
        else
            Z = A*Y;
        end

        if mod(j,2) == 0
            Q = orth(Z);
        end
    end    
    Q = orth(Z);

    %B = Q'*A; % k \times m * m \times n = k \times n
    %Bt = B'; % n \times k
    Bt = A'*Q;

    [Qhat,Rhat] = qr(Bt,'0');

    % Rhat is k \times k
    whos Qhat Rhat

    [Uhat,Sigmahat,Vhat] = svd(Rhat);

    U = Q*Vhat;
    Sigma = Sigmahat;
    V = Qhat*Uhat;
end
