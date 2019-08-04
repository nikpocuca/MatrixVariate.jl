module MatrixVariate
print("    __  ___ _    __
   /  |/  /| |  / /
  / /|_/ / | | / /
 / /  / /  | |/ /
/_/  /_/   |___/
                                      \n")
	import LinearAlgebra, Rmath ,RCall ,StatsBase, HypothesisTests, DataStructures, JLD2, Distances, Statistics, Random, ProgressMeter

	export MatrixVariateTest, MatrixVariateModel, MatVTest, ddplot, MatrixVariateNormalTest, MatrixVariateTest, mkmeans, mbiclust

	using HypothesisTests, RCall, JLD2

	# ========================================
	# Version 0.1.0 +
	# ========================================

	# MatrixVariate Test
	include("MatrixNormTest/MatrixNormTest.jl")

	# ========================================
	# Version 0.2.0 +
	# ========================================

	# Matrix Bilinear Facor Analyzers
	include("MbiClust/mbiclust.jl")

	# Matrix Kmeans
	include("KMeans/MatrixKMeans.jl")

	# mvgen, and mix_gen, generates matrix variate data
	include("MVNGen/mvngen.jl")
	include("MVNGen/mix_gen.jl")

	# ========================================

	# Abstract type definitions
	abstract type MatrixVariateTest end
	abstract type MatrixVariateModel end

end # module
