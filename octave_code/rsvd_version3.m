% QR of B^T version with sampling of (A A^T)^q A R matrix with orthogonalization 
% control parameter s (s=1 most paranoid, s>1 less paranoid). 
function [U,Sigma,V] = rsvd_version3(A,k,q,s)
    m = size(A,1);
    n = size(A,2);

    R = randn(n,k);
    Y = A*R; % m \times n * n \times k = m \times k

    for j=1:q
        if mod(2*j-2,s) == 0
            [Y,~] = qr(Y,'0');
        end
        Z = A'*Y;

        if mod(2*j-1,s) == 0
            [Z,~] = qr(Z,'0');
        end
        Y = A*Z;
    end    
    [Q,~] = qr(Y,'0');


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
