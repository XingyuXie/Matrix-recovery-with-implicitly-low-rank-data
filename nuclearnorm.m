function nnorm = nuclearnorm( X )

% compute the nuclear norm of a matrix.

s = svd(X) ;
nnorm = sum( s ) ;
