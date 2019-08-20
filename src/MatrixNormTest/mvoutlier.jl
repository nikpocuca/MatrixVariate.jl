using LinearAlgebra

# Computes the Matrix Variate Mahalanobis Squared Distance
# X_i is observation
# M_i is mean matrix
# U_i is row Co-variance
# V_i is col Co-variance


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


# Computes the Multivariate Mahalanobis Squared Distance
# X_i is observation
# μ is mean matrix
# Σ is covariance matrix


function mul_dist(X_i::Array{Float64,3},
		  μ::Array{Float64,1},
		  Σ::Array{Float64,2})::Array{Float64,1}

	# reshape X_i into vector form.
	d::Int64 = size(X_i,1);
	N::Int64 = size(X_i,3);
	x_i::Array{Float64,2} = Array{Float64,2}(reshape(X_i,(d^2,N)))';

	# Invert Covariance Matrix
	Σ_inv::Array{Float64,2}  = inv(Σ);

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
