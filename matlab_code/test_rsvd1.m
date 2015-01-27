%make_matrix2;

more off;
fprintf('loading..\n');
%load('../data/A_mat1.mat');
load('data/A_mat1.mat');
whos M

k=500;
fprintf('compute rsvd with k=%d\n', k);
tic;
%[U,Sigma,V] = rsvd_version1(A,k);
%[U,Sigma,V] = rsvd_version2(A,k);
[U,Sigma,V] = rsvd_version3(A,k,3,1);
elapsed_time = toc();
fprintf('done in %f sec\n', elapsed_time);

Ak = U*Sigma*V';
percent_error = 100*norm(A - Ak,'fro')/norm(Ak,'fro')

