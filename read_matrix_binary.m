% read matrix from bin file
function M = read_matrix_binary(bin_file)

% read file
fprintf('reading matrix from %s\n', bin_file);
fp = fopen(bin_file,'r');
m = fread(fp,1,'int32');
n = fread(fp,1,'int32');
M = zeros(m,n);
for i=1:m
    if mod(i,100) == 0
        fprintf('reading row %d of %d\n', i,m);
    end
    for j=1:n
        M(i,j) = fread(fp,1,'double');
    end
end
fclose(fp);
