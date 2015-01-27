% Version II with autorank 1 
% frac_of_max_rank : what fraction of min(m,n) to use; must be <=1
% TOL : when to quit when norm of projection vector < TOL
function [U,Sigma,V] = rsvd_version2_auto_rank1(A,frac_of_max_rank,TOL)
    m = size(A,1);
    n = size(A,2);
    maxdim = round(frac_of_max_rank*min(m,n));

    R = randn(n,maxdim);
    Y = A*R; % m \times n * n \times k = m \times k

    fprintf('running Gram Schmidt to find suitable rank..\n');
    Qbig = Y;
    clear Y;
    forbreak = 0;
    p1 = zeros(maxdim,1);

    for j=1:maxdim
        estimated_rank = j;
        ej = zeros(maxdim,1);
        ej(j) = 1;

        vj = Qbig*ej;
        for i=1:(j-1)
            ei = zeros(maxdim,1);
            ei(i) = 1;
            vi = Qbig*ei;
            p = project_vec(vj,vi);
            vj = vj - p;
            if norm(p) < TOL && norm(p1) < TOL
                forbreak = 1;
                break;
            end
            p1 = p;
        end
        vj = vj/norm(vj);
        Qbig(:,j) = vj;
        if forbreak == 1
            break;
        end
    end

    Q = Qbig(:,1:estimated_rank);
    [Q,~] = qr(Q,0);
    fprintf('using Q of size %d times %d\n', size(Q,1), size(Q,2));

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

