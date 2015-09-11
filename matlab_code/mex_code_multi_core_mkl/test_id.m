m = 5000;
n = 7000;
k = 3000;

fprintf('making %d x %d matrix A..\n',m,n);
p = min(m,n);
if m >= n
   [U, temp] = qr(randn(m,n),0);
   [V, temp] = qr(randn(n));
else
   [U, temp] = qr(randn(m));
   [V, temp] = qr(randn(n,m),0);
end
if p>1
    %S = logspace(1,-5,p);
    S = logspace(1,-3,p);
else
    S = [1];
end
S = diag(S);
A = U*S*V';

fprintf('running ID of rank %d via mex file..\n', k);
[I,T] = id_mkl_mex_interface1(A,k);

whos I T

P = zeros(n,n);
for j=1:n
    P(I(j),j) = 1;
end

Id = eye(k,k);
Ts = T';
V = [Id; Ts];
Vt = V';
VP = P*V;
VPt = VP';

A_approx = A(:,I(1:k))*Vt;
A_approx_with_P = A(:,I(1:k))*VPt;

fprintf('percent errors:\n');
norm(A(:,I) - A_approx)/norm(A) * 100
norm(A - A_approx_with_P)/norm(A) * 100

