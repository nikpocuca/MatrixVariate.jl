module MatrixNormTest

	import LinearAlgebra, StatsBase, HypothesisTests, DataStructures, JLD2

	export MatVTest, ddplot, MatrixVariateNormalTest, MatrixVariateTest

	using HypothesisTests, RCall, JLD2


	# Matrix Variate Mahalanobis Distance functions
	include("mvoutlier.jl")
	# Maximum Likelihood Estimates
	include("mvMLE.jl")
	include("multiMLE.jl")

	print("Matrix Normality Test\n")

	abstract type MatrixVariateTest end

	mutable struct MatrixVariateNormalTest
		d_mat::Array{Float64,1};
		d_mult::Array{Float64,1};
		ks_test::ApproximateTwoSampleKSTest;
		p_val::Float64
	end

	# MAIN Matrix Variate Normal Test function.
	# X is an r x c x N array.
	# α is power level.
	# iter is the number of mle iterations for the matrix variate MLE estimate.
	"""
		MatVTest(X::Array{Float64,3};α::Float64 = 0.05,iter::Int64 = 100)

	"""
	function MatVTest(X::Array{Float64,3};α::Float64 = 0.05,iter::Int64 = 100)::MatrixVariateNormalTest

		     mdl::OrderedDict{Symbol,Any} = mvMLE(X,iter);

		     # get estimates from the model
		     M::Array{Float64,2} = mdl[:M][:,:,1];
		     U::Array{Float64,2} = mdl[:U];
		     V::Array{Float64,2} = mdl[:V];

		     # Compute distances
		     distances_Matrix::Array{Float64,1} = mvo_dist(X,M,U,V);

		     # vectorize X
		     mdl2::OrderedDict{Symbol,Any} = mlestimate(X)

		     μ::Array{Float64,1} = mdl2[:mu][:,1]
		     σ::Array{Float64,2} = mdl2[:sigma]

		     distances_multi::Array{Float64,1} = mul_dist(X,μ,σ)

			 kstest = ApproximateTwoSampleKSTest(distances_Matrix,distances_multi);

			 pval::Float64 = pvalue(kstest);

			 return MatrixVariateNormalTest(distances_Matrix,distances_multi,kstest,pval) ;
	end


	## DD plot, takes in a mat_test and plots it using R
	"""
		ddplot(mat_test::MatrixVariateNormalTest)

	"""
	function ddplot(mat_test::MatrixVariateNormalTest)::nothing
		# declare distances
		d_mat = mat_test.d_mat;
		d_mult = mat_test.d_mult;
		@rput d_mat d_mult
		R"
		plot(d_mat,
		d_mult,type = 'p', pch = 20,
			cex = 0.5,xlab = 'Matrix Variate MSD',ylab = 'Multivariate MSD')
		abline(0,1,col = 'red')
		rm(d_mat)
		rm(d_mult)
		"
	end


	function example_data()
		basename = joinpath(dirname(@__FILE__), "data","ex.jld2")::String
		@load basename X_A X_B
		return(X_A,X_B)
	end

end # module
