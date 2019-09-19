function obj = cal_obj(X, A, KG, rho, opt)

%[~,n] = size(X);
%K = constructKernel(X',A',opt);
%obj1 = n-trace(K);
obj1 = sum(sqrt(diag((X-A)'*(X-A))));

K_A = constructKernel(A',A',opt);
obj2 = norm(KG-K_A,'fro')*norm(KG-K_A,'fro');
%obj2 = sum(sum(abs(KG-K_A)));

obj = obj1 + 0.5*rho*obj2;
