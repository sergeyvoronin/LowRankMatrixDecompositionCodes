more off;
m = 50; n = 50;
A = randn(m,n);
AtA = A'*A;
A = 0.5*(AtA + AtA');

save('data/A_mat1.mat','A');
