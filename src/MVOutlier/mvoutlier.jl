
#Detection for outlier functions 
# Written by Nik Pocuca 

using LinearAlgebra

# Computes the Matrix Variate Mahalanobis Distance 
# X_i is observation 
# M_i is mean matrix 
# U_i is row Co-variance 
# V_i is col Co-variance 

function mvodist(X_i::Array{Float64,2},M_i::Array{Float64,2},U_i::Array{Float64,2},V_i::Array{Float64,2})::Float64 
#=	
	try
		# Compute Cholesky Decomposition 
		UL = cholesky(U_i).L
		VL = cholesky(V_i).L
		print("Hitting cholesky")	
		# Invert Lower triangular matrices
	
		I_UL = inv(UL)
		I_VL = inv(VL)

		Uinv = (I_UL')*I_UL
		Vinv = (I_VL')*I_VL
	catch

=#
		Uinv = inv(U_i)
		Vinv = inv(V_i)
#	end

		# Distribution object of standard normal 
		XM = X_i - M_i
		d_XM = tr( Uinv*(XM)*Vinv*(XM') )
		#d_XM = tr(inv(U_i)*(XM)*inv(V_i)*(XM'))

	return d_XM
end




function muldist(xi,mu,Sig)

	#Compute invertable sig 


	Siginv = inv(Sig)
	
	xm = (xi - mu)'
	dm_h = (xm)*Siginv*xm'
	return(dm_h)
end




# Same functions but for large scale computations. 
function mvo_dist(X_i::Array{Float64,3},
		  M_i::Array{Float64,2},
		  U_i::Array{Float64,2},
		  V_i::Array{Float64,2})::Array{Float64,1} 

	# invert covariance matrices once. 
	U_inv::Array{Float64,2} = inv(U_i);
	V_inv::Array{Float64,2} = inv(V_i); 
	
	# initialize distances 	
	N = size(X_i,3);
	distances::Array{Float64,1} = Array{Float64,1}(undef,N);
	
	# Begin computing distances  
	@simd for i=1:N
		
		@inbounds local XM::Array{Float64,2} = Array{Float64,2}(X_i[:,:,i] - M_i);

		local d_i::Float64 = tr(U_inv*XM*V_inv*XM');
	
		@inbounds distances[i] = d_i;  
	end 

	return distances; 
end


function mul_dist(X_i::Array{Float64,3},
		  μ::Array{Float64,1},
		  Σ::Array{Float64,2})::Array{Float64,1}

	# reshape X_i into vector form. 
	d::Int64 = size(X_i,1);
	N::Int64 = size(X_i,3); 
	x_i::Array{Float64,2} = Array{Float64,2}(reshape(X_i,(d^2,N)))'; 

	# Invert Covariance Matrix
	Σ_inv  = inv(Σ);

	# Convert distances 	 
	distances::Array{Float64,1} = Array{Float64,1}(undef,N);

	# Begin computing distances 
	@simd for i=1:N

		local xm::Array{Float64,1} = Array{Float64,1}(x_i[i,:] - μ); 
		local d_i::Float64 = (xm)'*Σ_inv*(xm);	
		
		@inbounds distances[i] = d_i; 
	end
	
	return distances; 
end
