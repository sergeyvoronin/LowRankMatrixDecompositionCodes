%make_matrix2;

more off;
fprintf('loading..\n');
%load('../data/A_mat1.mat');
load('data/A_mat2.mat');
whos M

k=100;
fprintf('compute rsvd with k=%d\n', k);
tic;
%[U,Sigma,V] = rsvd_version2(M,k);
%[U,Sigma,V] = rsvd_version3_old(M,k,3);
[U,Sigma,V] = rsvd_version3(M,k,3,4);
elapsed_time = toc();
fprintf('done in %f sec\n', elapsed_time);

P = U*Sigma*V';
percent_error = 100*norm(M - P,'fro')/norm(M,'fro')

