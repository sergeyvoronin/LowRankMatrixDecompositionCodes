more off;
fprintf('loading..\n');
load('data/A_mat1.mat');
whos M

ks=[100:10:500];

elapsed_times1 = zeros(length(ks),1);
elapsed_times2 = zeros(length(ks),1);
elapsed_times3 = zeros(length(ks),1);


percent_errors1 = zeros(length(ks),1);
percent_errors2 = zeros(length(ks),1);
percent_errors3 = zeros(length(ks),1);

for i=1:length(ks)
    k = ks(i);

     
    fprintf('compute rsvd1 with k=%d\n', k);
    tic;
    [U,Sigma,V] = rsvd_version1(A,k);
    elapsed_time = toc();
    fprintf('done in %f sec\n', elapsed_time);
    Ak = U*Sigma*V';
    percent_error = 100*norm(A - Ak,'fro')/norm(Ak,'fro')
    elapsed_times1(i) = elapsed_time;
    percent_errors1(i) = percent_error;

    
    fprintf('compute rsvd2 with k=%d\n', k);
    tic;
    [U,Sigma,V] = rsvd_version2(A,k);
    elapsed_time = toc();
    fprintf('done in %f sec\n', elapsed_time);
    Ak = U*Sigma*V';
    percent_error = 100*norm(A - Ak,'fro')/norm(Ak,'fro')
    elapsed_times2(i) = elapsed_time;
    percent_errors2(i) = percent_error;



    fprintf('compute rsvd3 with k=%d\n', k);
    tic;
    [U,Sigma,V] = rsvd_version3(A,k,3,1);
    elapsed_time = toc();
    fprintf('done in %f sec\n', elapsed_time);
    Ak = U*Sigma*V';
    percent_error = 100*norm(A - Ak,'fro')/norm(Ak,'fro')
    elapsed_times3(i) = elapsed_time;
    percent_errors3(i) = percent_error;

   
end

save('data/run1.mat');
