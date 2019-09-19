function [Ak,out] = solve_A2(X, KB, rho, opts)
% solve the following problem
% \min_A \|X-A\|_{2,1} + rho/2 \|KB - \pha(A)^T * pha(A)\|_F^2
% Author: Xingyu Xie, 2017.10.26

%% Parameters and defaults
if isfield(opts,'tol'),    tol = opts.tol;     else tol = 1e-4;   end
if isfield(opts,'maxit'),  maxit = opts.maxit; else maxit = 500;  end
%if isfield(opts,'maxitA'),  maxit = opts.maxitA; else maxitA = 50;  end
if isfield(opts,'maxT'),   maxT = opts.maxT;   else maxT = 1e3;   end

%% Kernel Options
options.KernelType = 'Gaussian';
if isfield(opts,'gam'),     options.gam =opts.gam;       else  options.gam = 0.22;       end

%% Data preprocessing and initialization
[~,n] = size(X);
start_time = tic; gamma = opts.gam;
%fprintf('Iteration:     ');
LA_idx = 2*gamma*rho*n;
alpha_idx = 0.9/LA_idx ;
%% Iterations of block-coordinate update 
if isfield(opts,'Ak0'), Ak0 = opts.Ak0; else Ak0 = X; end
Ak = Ak0; Zk = Ak;Vk = Ak;
Yk = Ak;
nstall = 0; t0 = 1;
obj0 = cal_obj(X, Ak, KB, rho, options);
%for p0_idx = 1:maxitA
%P0 = diag(constructKernel(X',Ak',options));

for k = 1:maxit
    fprintf('\b\b\b\b\b%5i',k);   
    A_last = Ak;
    Y_last = Yk;
     % --- A-update ---
    for idx = 1:n
        %p0 = P0(idx);
        m = KB(:,idx);


        %K_A2 = constructKernel(A_last',Ak(:,idx)',tempOpt);
        %tmp2_coff = - 2*rho*gamma.*K_A2;
        %tmp3_coff = 2*rho*gamma.*m.*constructKernel(A_last',Ak(:,idx)',options);
        exp_vec = constructKernel(A_last',Ak(:,idx)',options);
        tmp3_coff = - 2*gamma*rho.*(exp_vec - m).*exp_vec;
        %tmp2 = (repmat(Ak(:,idx),[1,n]) - A_last)*(tmp2_coff+tmp3_coff);
        tmp2 = (repmat(Ak(:,idx),[1,n]) - A_last)*(tmp3_coff);

        temp_ak = Ak(:,idx) - alpha_idx.*tmp2;
        temp_ak = temp_ak - X(:,idx);
        vk = solve_l1l2(temp_ak,alpha_idx);
        %vk = max(0,temp_ak - alpha_idx)+min(0,temp_ak + alpha_idx);
        vk = vk + X(:,idx);
        %vk = (4*gamma*p0.*X(:,idx)+1.0/alpha_idx * temp_ak)./(4*gamma*p0 + 1.0/alpha_idx);
        Vk(:,idx) = vk;
        A_last(:,idx) = vk;
        %obj_vk = cal_line_test(xk, X_cln, vk, m, rho, p0, options);


        %K_A2 = constructKernel(Y_last',Yk(:,idx)',tempOpt);
        %tmp2_coff = - 2*rho*gamma.*K_A2;
        %tmp3_coff = 2*rho*gamma.*m.*constructKernel(Y_last',Yk(:,idx)',options);
        %tmp2 = (repmat(Yk(:,idx),[1,n]) - Y_last)*(tmp2_coff+tmp3_coff);
        exp_vec = constructKernel(Y_last',Yk(:,idx)',options);
        tmp3_coff = - 2*gamma*rho.*(exp_vec - m).*exp_vec;
        tmp2 = (repmat(Yk(:,idx),[1,n]) - Y_last)*(tmp3_coff);
        
        temp_ak = Yk(:,idx) - alpha_idx.*tmp2;
        temp_ak = temp_ak - X(:,idx);
        %zk = (4*gamma*p0.*X(:,idx)+1.0/alpha_idx * temp_ak)./(4*gamma*p0 + 1.0/alpha_idx);
        zk =  solve_l1l2(temp_ak,alpha_idx);
        %zk = max(0,temp_ak - alpha_idx)+min(0,temp_ak + alpha_idx);
        zk = zk + X(:,idx);
        Zk(:,idx) = zk;
        Y_last(:,idx) = zk;
        %obj_zk = cal_line_test(xk, X_cln, zk, m, rho, p0, options);
    end
    % --- diagnostics, reporting, stopping checks ---
    obj_vk = cal_obj(X, Vk, KB, rho, options);%cal_obj_sub3(X, P0, Vk, KB, rho, options);
    obj_zk = cal_obj(X, Zk, KB, rho, options);%cal_obj_sub3(X, P0, Zk, KB, rho, options);
    if obj_zk>=obj_vk 
        % restore to previous A, and cached quantities for nonincreasing objective
        Ak = Vk;
    else
        % extrapolation
        Ak = Zk;
    end
        % reporting
    obj = min(obj_vk,obj_zk);%cal_obj(X, Ak, KB, rho, options);
    out.hist_obj(k) = obj;
    out.relerr1(k) = abs(obj-obj0)/(obj0+1);
    out.relerr2(k) = max(max(abs(Ak-Ak0)));
    obj0 = obj;

    % stall and stopping checks
    crit = (out.relerr1(k)<tol);
    if crit; nstall = nstall+1; else nstall = 0; end
    if nstall >= 3 || out.relerr2(k) < tol, break; end
    if toc(start_time) > maxT; break; end;
    
    

    % --- correction and extrapolation ---
    t = (1+sqrt(1+4*t0^2))/2;
    w2 = (t0-1)/t;
    w1 = t0/t;
    Yk = Ak + w1*(Zk-Ak) + w2*(Ak-Ak0); % extrapolation

    Ak0 = Ak; t0 = t; 
%end

end

out.iter = k;
fprintf('\n');  % report # of iterations
