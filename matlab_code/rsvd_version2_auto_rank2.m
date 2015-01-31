% Version II with autorank II
% kblock: size of random samples block added at each update
% TOL: when to stop adding samples, when ||Q*Q'*A - A||_2 < TOL
function [U,Sigma,V] = rsvd_version2_auto_rank2(A,kblock,TOL)
    m = size(A,1);
    n = size(A,2);

    R = randn(n,kblock);
    Y = A*R; % m \times n * n \times k = m \times k

    exit_loop = 0;
    while exit_loop~=1
        %Q = qr(Y,0); % m \times k
        [Q,~]=qr(Y,0);
        norm_tol = norm(Q*Q'*A - A)/norm(A);
        if norm_tol > TOL
            fprintf('norm_tol = %f; increase rank..\n', norm_tol);
            R = randn(n,kblock);
            Y_new = A*R; % m \times n * n \times k = m \times k
            Y = [Y, Y_new];
        else
            exit_loop = 1;
        end
    end

    fprintf('using Q of size %d \times %d\n', size(Q,1), size(Q,2));

    %B = Q'*A; % k \times m * m \times n = k \times n
    Bt = A'*Q;

    [Qhat,Rhat] = qr(Bt,0);

    % Rhat is k \times k
    whos Qhat Rhat

    [Uhat,Sigmahat,Vhat] = svd(Rhat);

    U = Q*Vhat;
    Sigma = Sigmahat;
    V = Qhat*Uhat;
end
