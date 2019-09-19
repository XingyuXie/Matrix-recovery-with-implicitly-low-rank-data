function obj = cal_obj_sub3(X, P0, A, KG, rho, opt)

%[~,n] = size(X);
K_term1 = P0.*diag((A-X)'*(A-X));
obj1 = sum(K_term1);

K_A = constructKernel(A',A',opt);
obj2 = norm(KG-K_A,'fro')*norm(KG-K_A,'fro');

obj = 2*opt.gam*obj1 + 0.5*rho*obj2;
