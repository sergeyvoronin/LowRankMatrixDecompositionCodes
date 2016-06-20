% test the different rsvd algorithms
make_matrix = 1;
save_matrix = 0;

if make_matrix == 1
    fprintf('making matrix..\n');
    m = 3000; n = 1000; p = min(m,n);
    S = logspace(0,-2,p);
    S = diag(S);
    mat_filename = ['data/M_mat1.mat']
    M = make_matrix1(m,n,S,mat_filename,save_matrix);
else
    load('data/M_mat1.mat');
    m = size(M,1);
    n = size(M,2);
end

% set params
k = 300;
p = 100;
q = 2;
s = 1;

fprintf('rsvd 1..\n');
[U,Sigma,V] = rsvd_version1(M,k,p,q,s);
whos M U Sigma V
perror = norm(M - U*Sigma*V')/norm(M) * 100

fprintf('rsvd 2..\n');
[U,Sigma,V] = rsvd_version2(M,k,p,q,s);
whos M U Sigma V
perror = norm(M - U*Sigma*V')/norm(M) * 100

fprintf('rsvd 3..\n');
kstep = 20;
[U,Sigma,V] = rsvd_version3(M,k,kstep,q,s);
perror = norm(M - U*Sigma*V')/norm(M) * 100

