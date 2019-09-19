load('mnist.mat')
images = training.images;
labels = training.labels;
X =images(:,:,labels == 3);
% X = zeros([14,14,5958]);
% for i = 1:5958
%     X(:,:,i) = imresize(X_big(:,:,i), 0.5);
% end
X = reshape(X,[784,6131]);
X = X(:,1:400);

%rotate
for n=1:400
    RotateDgree=randi([-90 90]);
    ts=X(:,n);
    ts = reshape(ts,[28,28]);
    ts=imrotate(ts,RotateDgree,'nearest','crop');
    ts = reshape(ts,[784,1]);
    X(:,n)= ts;
end


[D, nX] = size(X);%784*400
s2 = sum(X.^2,1);
s2 = sqrt(s2);
s2max = max(s2);
X = X.*repmat(1./s2,784,1);
X_cln = X;
inds = rand(1,nX)<=0.1;
noise_factor = 0.05;
noise = noise_factor.*randn(D,nX);

X(:,inds) = X(:,inds) + noise(:,inds);
%X(X>(1.0/s2max)) = (1.0/s2max);
X(X<0.0) = 0.0;
%X = reshape(X,[28,28,400]);




options.KernelType = 'Gaussian';
options.d =  1.0;
options.gam = 0.9;
KB = constructKernel(X_cln', X_cln',options);

%options.KernelType = 'Linear';

KX = constructKernel(X', X',options)*0.85;
rho = 13;
lambda = 0.5;
opts.gam = 1.0;
opts.tol = 1e-6;
opts.ker  = 1;
opts.initlam = 0.95;
opts.K = KX;
opts.inittype = 2;
[ A, B, out ] = KRPCA(X,lambda, rho, opts );

lambda = 0.8; % [0,1] 0.5(+lowrank) 0.4 0.3 0.2(+sparse)
OMG = ones(784,400);
[L,S] = mr_pca_part(X,OMG,lambda);

norm(X_cln-L,'fro')/norm(X_cln,'fro')
norm(X_cln-A,'fro')/norm(X_cln,'fro')
norm(X_cln-X,'fro')/norm(X_cln,'fro')
rank(B,norm(B,2)*1e-4)

PSNR = 0.0;
SNR = 0.0;
for i = 1:400
    %if inds(i)>0
        t = A(:,i)*s2(i);
        t = reshape(t,[28,28]);
        t(t<0.1)  = 0.0;
        [peaksnr, snr] = psnr(t,reshape(X_cln(:,i),[28,28])*s2(i));
        PSNR = PSNR + peaksnr;
        SNR = SNR + snr;
    %end        
end

PSNR2 = 0.0;
SNR2 = 0.0;
for i = 1:400
    %if inds(i)>0
        t = L(:,i)*s2(i);
        t = reshape(t,[28,28]);
        [peaksnr, snr] = psnr(t,reshape(X_cln(:,i),[28,28])*s2(i));
        PSNR2 = PSNR2 + peaksnr;
        SNR2 = SNR2 + snr;
   % end
end
PSNR/400
PSNR2/400
SNR/400
SNR2/400
