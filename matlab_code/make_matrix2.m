more off;

% singular values fall of more, more ill-conditioned example 
m = 1000;
n = 1500;

fprintf('making %d x %d matrix\n', m,n);

p = min(m,n);
if m >= n
   [U, temp] = qr(randn(m,n),0);
   [V, temp] = qr(randn(n));
else
   [U, temp] = qr(randn(m));
   [V, temp] = qr(randn(n,m),0);
end
if p>1
    S = logspace(0,-5,p);
else
    S = [1];
end
S = diag(S);
A = U*S*V';

save('data/A_mat2.mat','A','S');

