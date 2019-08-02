module MatrixVariate

	import LinearAlgebra, StatsBase, HypothesisTests, DataStructures, JLD2

	export MatVTest, ddplot, MatrixVariateNormalTest, MatrixVariateTest

	using HypothesisTests, RCall, JLD2

	include("MatrixNormTest/MatrixNormTest.jl")

	print("Matrix Normality Test\n")

	# abstract type definitions
	abstract type MatrixVariateTest end

end # module
