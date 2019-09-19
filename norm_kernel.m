function [ M ] = norm_kernel( K )

if ~issymmetric(K)
    error(['In ''' mfilename ''': affinity matrix is not symmetric'])
end

[m,~] = size(K);
M = zeros(m,m);
for i = 1:m
    for j = 1:m
        if K(i,j)~=0
         M(i,j) = K(i,j)/sqrt(K(i,i)*K(j,j));
        end
    end
end



end

