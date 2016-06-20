% QR of B^T version 
function [U,Sigma,V] = rsvd_version2(A,k)
    m = size(A,1);
    n = size(A,2);

    R = randn(n,k);
    Y = A*R; % m \times n * n \times k = m \times k

    [Q,~] = qr(Y,0); % m \times k

    %B = Q'*A; % k \times m * m \times n = k \times n
    %Bt = B'; % n \times k
    Bt = A'*Q;

    [Qhat,Rhat] = qr(Bt,0);

    % Rhat is k \times k
    whos Qhat Rhat

    [Uhat,Sigmahat,Vhat] = svd(Rhat);

    U = Q*Vhat;
    Sigma = Sigmahat;
    V = Qhat*Uhat;
end
