# Compute the unbiased maximum likelihood estimates of mu vector and sigma of a multivariate normal.

using StatsBase, DataStructures, Statistics

function unbi_mult_estimate(X::Array{Float64,3})::OrderedDict{Symbol,Any}

	# compute map slices
	x_is::Array{Float64,3} = mapslices(vec,X,dims=[1,2]);

	# compute estimate of mu_hat
	mu_hat::Array{Float64,2} = mapslices(Statistics.mean,x_is,dims=3)[:,:,1];

	#estimate sigma
	inner_terms::Array{Float64,3} = mapslices(x -> ((x-mu_hat))*(x-mu_hat)',x_is,dims = [1,2]);
	sigma::Array{Float64,3} = mapslices(sum,inner_terms,dims=3)/((size(x_is,3))-1);

	rez = OrderedDict{Symbol,Any}();

	rez[:mu] = mu_hat
	rez[:sigma] = sigma[:,:,1]

	return(rez)
end
