# Matrix Variate Normal Unbiased Estimate based on Duthiell
using LinearAlgebra


function unbi_mat_estimate(X::Array{Float64,3},maxiter::Int64=100)::OrderedDict{Symbol,Any}

    N::Int64 = size(X,3);
    r::Int64 = size(X,1);
    c::Int64 = size(X,2);

    # Return var
    returnVar::OrderedDict{Symbol,Any} = OrderedDict{Symbol,Any}();

    # estimate mean
    M::Array{Float64,2} = (mapslices(sum,X, dims=3)/N)[:,:,1];

    # Initilize U and V
    V_eye::Array{Float64,2} = (zeros(c,c) + I);

    U_o::Array{Float64,2} = U_est(X,M,V_eye);
    V_o::Array{Float64,2} = V_est(X,M,U_o);

    U::Array{Float64,2} = U_o;
    V::Array{Float64,2} = V_o;

    for i = 1:maxiter
        # Estimate U then V
        U = U_est(X,M,V)
        V = V_est(X,M,U)
    end

    returnVar[:M] = M
    returnVar[:U] = U
    returnVar[:V] = V
    return returnVar
end

function U_est(X::Array{Float64,3},
		M::Array{Float64,2},
		V::Array{Float64,2})::Array{Float64,2}

    N::Int64 = size(X,3);
    c::Int64 = size(X,2);
    r::Int64 = size(X,1);

    # Check if V is invertible
    # Invert V matrix and compute cholesky
    V_inv::Array{Float64,2} = inv(V)

    U::Array{Float64,2} = zeros(r,r);

    # Begin for loop

    denom::Float64 = (N-1)*c;
    for i::Int64 = 1:N # Through observations
        # Using cholesky variant instead of direct computation for symmetry
        #L = V_inv_chol[:,:,j][1].factors';
        XML::Array{Float64,2} = (X[:,:,i] - M)[:,:,1] #*L;
        U += (XML*V_inv*(XML)')/denom;
    end

    return(U);
end



function V_est(X::Array{Float64,3},
		M::Array{Float64,2},
		U::Array{Float64,2})::Array{Float64,2}

    N::Int64 = size(X,3);
    c::Int64 = size(X,2);
    r::Int64 = size(X,1);

    # Check if V is invertible
    # Invert V matrix and compute cholesky
    U_inv::Array{Float64,2} = inv(U);

    V::Array{Float64,2} = zeros(c,c);

    # Begin for loop

    denom::Float64 = N*r;
    for i::Int64 = 1:N # Through observations
        # Using cholesky variant instead of direct computation for symmetry
        #L = V_inv_chol[:,:,j][1].factors';
        XML::Array{Float64,2} = (X[:,:,i] - M)[:,:,1]#*L;
        V += ((XML')*U_inv*XML)/denom;
    end

    return(V);
end
