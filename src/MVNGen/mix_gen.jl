


# Generates mixtures of matrix variate normal 

function mix_mvgen(k::Int64,N::Int64,d::Int64)::Array{Float64,3}

	# Initialize Multidimensional Array 
	XPS = Array{Float64,3}(undef,d,d,N*k); 

	# Start for loop for each group. 
	for i=1:k
	
		local M = Array{Float64,2}(rand(d,d)*i);
		local u = Array{Float64,2}(rand(d,d).*sqrt(i)); 
		local v = Array{Float64,2}(rand(d,d).*sqrt(i)); 
		local U = (u*u'/2); local V = v*v'; 	
		
		XS_i::Array{Float64,3} = MVMix.mvngen(N,M=M,U_in=U,V_in=V);
		@inbounds XPS[:,:,((1+N*(i-1)):(N*i))] = XS_i; 

	end # ends for loop

	return XPS;
 
end # ends function




# Generates Mixtures of multivariate normal but no kronecker 

function mix_multigen(k::Int64,N::Int64,d::Int64)::Array{Float64,3}

	# Initialize Multidimensional array of appropriate size 
	local XPS = Array{Float64,3}(undef,d,d,N*k); 
	
	# for each group... 
	for i=1:k

		local mu = Array{Float64,1}(rand(d^2)*i); 
		local sigma = Array{Float64,2}(rand(d^2,d^2)*sqrt(i))
		local Sigma::Array{Float64,2} = sigma*sigma'/2;
		
		local xs_i::Array{Float64,2} = MVMix.multigen(N,mu,Sigma);
		
		# Reshape sticks 
		local XS_i = Array{Float64,3}(reshape(xs_i',(d,d,N)));
	 
		@inbounds XPS[:,:,((1+N*(i-1)):(N*i))] = XS_i; 

	end # ends for loop 
 	
	return XPS;
 
end # ends function 




