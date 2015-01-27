%make_matrix2;

more off;
fprintf('loading..\n');
%load('../data/A_mat1.mat');
load('data/A_mat1.mat');
whos M

fprintf('compute rsvd with autorank\n');
tic;
%[U,Sigma,V] = rsvd_version2_auto_rank1(A,0.5,0.001);
[U,Sigma,V] = rsvd_version2_auto_rank2(A,100,0.1);
elapsed_time = toc();
fprintf('done in %f sec\n', elapsed_time);

Ak = U*Sigma*V';
percent_error = 100*norm(A - Ak,'fro')/norm(Ak,'fro')

