
    using RecipesBase

	# Matrix Variate Mahalanobis Distance functions
	include("mvoutlier.jl")
	# Unbiased Estimates
	include("unbi_mult_estimate.jl")
	include("unbi_mat_estimate.jl")

	mutable struct MatrixVariateNormalTest <: MatrixVariateTest
		d_mat::Array{Float64,1};
		d_mult::Array{Float64,1};
		ks_test::ApproximateTwoSampleKSTest;
		p_val::Float64
	end


	"""
		MatVTest(X::Array{Float64,3};α::Float64 = 0.05,iter::Int64 = 100)

		# MAIN Matrix Variate Normal Test function.
		# X is an r x c x N array.
		# α is significance level.
		# iter is the number of iterations for the matrix variate estimates.
	"""
	function MatVTest(X::Array{Float64,3};α::Float64 = 0.05,iter::Int64 = 100)::MatrixVariateNormalTest

		mdl::OrderedDict{Symbol,Any} = unbi_mat_estimate(X,iter);

	 	# get estimates from the model
	 	M::Array{Float64,2} = mdl[:M][:,:,1];
		U::Array{Float64,2} = mdl[:U];
	 	V::Array{Float64,2} = mdl[:V];

		# Compute distances
		distances_Matrix::Array{Float64,1} = mvo_dist(X,M,U,V);

		# vectorize X
		mdl2::OrderedDict{Symbol,Any} = unbi_mult_estimate(X)

	 	μ::Array{Float64,1} = mdl2[:mu][:,1]
	   	σ::Array{Float64,2} = mdl2[:sigma]

 		distances_multi::Array{Float64,1} = mul_dist(X,μ,σ)

 		kstest = ApproximateTwoSampleKSTest(distances_Matrix,distances_multi);

 		pval::Float64 = pvalue(kstest);

 		return MatrixVariateNormalTest(distances_Matrix,distances_multi,kstest,pval) ;
	end

	# IO
	function Base.show(io::IO, test::MatrixVariateNormalTest)
	    println(io, "---------------------------")
	    println(io, "Matrix Variate Normal Test ")
	    println(io, "---------------------------")
		println(io, test.ks_test)
	end

 	# Plotting Tests

	@recipe function plot(m_test::MatrixVariateNormalTest)

		d_M::Array{Float64,1} = m_test.distances_Matrix;
		d_m::Array{Float64,1} = m_test.distances_multi;

		@series begin
	    	seriestype := :scatter
	    	color --> :black
			markersize = 2
	    	d_M, d_m
		end
	end


	## DD plot, takes in a mat_test and plots
	"""
		ddplot(mat_test::MatrixVariateNormalTest)

	"""
	function ddplot(mat_test::MatrixVariateNormalTest)::nothing
		# declare distances
		plot(mat_test)
		Plots.abline!(0,1)
	end

	"""
	example data function, loads an example for the ks test, see documentation
	# Call:
		example_data()
	"""
	function example_data()
		basename = joinpath(dirname(@__FILE__), "../data","ex.jld2")::String
		@load basename X_A X_B
		return(X_A,X_B)
	end
