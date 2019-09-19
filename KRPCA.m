function [ Ak, Bk, out ] = KRPCA(X, lambda ,rho, opts )
% solve the following problem
% \min_A,B \lambda\|B\|_* + \|\pha(X)-\pha(A)\|_F^2 + rho/2 \|B'*B - \pha(A)^T * pha(A)\|_F^2
% Author: Xingyu Xie, 2017.10.26

%% Parameters and defaults
if isfield(opts,'tol'),    tol = opts.tol;     else tol = 1e-4;   end
if isfield(opts,'maxitTol'),  maxitTol = opts.maxitTol; else maxitTol = 500;  end
if isfield(opts,'maxT'),   maxT = opts.maxT;   else maxT = 1e3;   end
if isfield(opts,'ker'),   ker = opts.ker;   else ker = 0;   end
%if isfield(opts,'rw'),     rw = opts.rw;       else rw = 1;       end

%% Kernel Options
options.KernelType = 'Gaussian';
if isfield(opts,'gam'),     options.gam =opts.gam;       else  options.gam = 0.22;       end

%% Data preprocessing and initialization
start_time = tic; 
fprintf('Iteration:     ');
%% Iterations of block-coordinate update 
KX = constructKernel(X',X',options);
%[m,n] = size(X);
[U,S,~] = svd(KX);
sigma = diag(S);
svp = length(find(sigma>sigma(1)*1e-2));
sigma = sigma(1:svp);
U = U(:,1:svp);
if opts.inittype == 1
      [d_X, n_X] = size(X);%1024*360
      OMG = ones(d_X,n_X);
      [Ak0,~] = mr_pca_part(X,OMG,opts.initlam);
%     opts.maxit = 7;
%     opts.maxitA = 7;
%     [Ak0,~] = solveA(X, K, rho, opts);
     opts.maxit = 1;
     opts.maxitA = 1;
elseif opts.inittype == 2
     opts.maxit = 7;
     opts.maxitA = 7;
     [Ak0,~] = solveA(X, opts.K, rho, opts);
     opts.maxit = 1;
     opts.maxitA = 1;
elseif inittype == 3
    opts.maxit = 50;
    [Ak0,~] = solve_A2(X, opts.K, rho, opts);
    opts.maxit = 1;
end 
opts.Ak0 = Ak0;
if isfield(opts,'Bk0'), Bk0 = opts.Bk0; else Bk0 = U*diag(sqrt(sigma))*U'; end
%if isfield(opts,'Ak0'), Ak0 = opts.Ak0; else Ak0 = zeros(m,n); end
Ak = Ak0; 
Bk = Bk0;
nstall = 0; 
KB = Bk'*Bk;
obj0 = lambda*nuclearnorm( Bk ) + cal_obj(X, Ak0, KB, rho, options);
for idx = 1:maxitTol
    fprintf('Total iter \b\b\b\b\b%5i  \n',idx);
    

    % Update B
    KA = constructKernel(Ak',Ak',options);
    Bk = solveB(KA, rho/lambda);
    KB =  Bk'*Bk;
    
        
    %Update A
    if ker
        [Ak,~] = solveA(X, KB, rho, opts);
    else
        [Ak,~] = solve_A2(X, KB, rho, opts);
    end
    opts.Ak0 = Ak;
    
    % --- diagnostics, reporting, stopping checks ---

    % reporting
    obj = lambda*nuclearnorm( Bk ) + cal_obj(X, Ak, KB, rho, options);
    out.hist_obj(idx) = obj;
    out.relerr1(idx) = abs(obj-obj0)/(obj0+1);
    out.relerr2(idx) = max([max(max(abs(Ak-Ak0))), max(max(abs(Bk-Bk0)))]);


    % stall and stopping checks
    crit = (out.relerr1(idx)<tol);
    if crit; nstall = nstall+1; else nstall = 0; end
    if nstall >= 3 || out.relerr2(idx) < tol, break; end
    if toc(start_time) > maxT; break; end;
    
    Ak0 = Ak; obj0 = obj; Bk0 = Bk;
%     rho = rho * 1.1;
%     lambda = lambda * 0.95;

out.iter = idx;
%fprintf('\n');  % report # of iterations
end
end

