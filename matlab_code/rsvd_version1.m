% BBt version 
function [U,Sigma,V] = rsvd_version1(A,k)
    m = size(A,1);
    n = size(A,2);

    R = randn(n,k); % this can be built up quickly on disk..
    Y = A*R; % m \times n * n \times k = m \times k

    Q = orth(Y);

    B = Q'*A;

    BBt = B*B';

    [Uhat,D] = eig(BBt);

    Sigma = sqrt(D);
    U = Q*Uhat;

    V = zeros(n,k);
    for j=1:k
        %v_j = 1/Sigma(j,j) * (A' * (Q * Uhat(:,j)));
        %v_j = 1/Sigma(j,j) * (B' * Uhat(:,j));
        v_j = 1/Sigma(j,j) * (A' * U(:,j));
        V(:,j) = v_j;
    end
end
