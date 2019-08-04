# Functions to Generate Matrix Variate Normals 

using StatsBase, ProgressMeter, DataStructures, RCall, Random, LinearAlgebra, Rmath

function mvngen(N;M,U_in,V_in)

	# Shape params
	r = size(M,1)
	c = size(M,2)
	# Check shape 
	if(size(M,1) != size(U_in,1))
		error("M row is not appropariate size for U")
	elseif(size(M,2) != size(V_in,1))
		error("M column is not appropriate size for V")
	end 
	
	# Generate Standard normals
	std_ns = rnorm(r*c*N)	
	
	Xs = reshape(std_ns,(r,c,N))
	
	A = cholesky(U_in).L
	B = cholesky(V_in).U

	X_e = mapslices(X -> M + A*X*B,Xs,dims=[1,2])

	return X_e 
end


# Generates multivariate normal through R
function multigen(N,mu,Sig)

	# checks for correct sizes 
	
	r = size(mu,1)
	rSig = size(Sig,1)
	
	if(r != rSig)
		error("parameters are not appropariate dimension")
	elseif(!isposdef(Sig))
		error("Sig is not positive definite")
	end	

	@rput mu Sig N
	R"library(MASS)"
	R"x = mvrnorm(n = N,mu = mu, Sigma = Sig)"
	@rget x 
	return x 
end



function mTests()
	
	# Example M matrix 
	Mt = zeros(2,2,3)
	Mt[:,:,1] = [1 3; 2 1]
	Mt[:,:,2] = [-1 0;4 -2]
	Mt[:,:,3] = [-2 2;5 5]
	
	U = zeros(2,2,3)
	# Identities 
	U[:,:,1] += I
	U[:,:,2] += I
	U[:,:,3] += I

	V = zeros(2,2,3)
	V[:,:,1] += I
	V[:,:,2] += I
	V[:,:,3] += I

	return Mt,U,V
end
