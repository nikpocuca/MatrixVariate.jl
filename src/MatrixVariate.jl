module MatrixVariate
print("    __  ___ _    __
   /  |/  /| |  / /
  / /|_/ / | | / /
 / /  / /  | |/ /
/_/  /_/   |___/
                                      \n")
	import LinearAlgebra, Rmath ,RCall ,StatsBase, HypothesisTests, DataStructures, JLD2, Distances, Statistics, Random, ProgressMeter

	export MatVTest, ddplot, MatrixVariateNormalTest, MatrixVariateTest, mkmeans, mbiclust

	using HypothesisTests, RCall, JLD2

	# MatrixVariate Test v. 0.1.0 +
	include("MatrixNormTest/MatrixNormTest.jl")
	# Matrix Bilinear Facor Analyzers  v. 0.2.0 +
	include("MbiClust/mbiclust.jl")
	# Matrix Kmeans v. 0.2.0 +
	include("KMeans/MatrixKMeans.jl")
	# mvgen, and mix_gen, generates matrix variate data v. 0.2.0 +
	include("MVNGen/mvngen.jl")
	include("MVNGen/mix_gen.jl")

	# abstract type definitions
	abstract type MatrixVariateTest end
	abstract type MatrixVariateModel end

end # module
