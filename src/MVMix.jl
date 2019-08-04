print("    __  ___ _    __ __  ___
   /  |/  /| |  / //  |/  /
  / /|_/ / | | / // /|_/ /
 / /  / /  | |/ // /  / /
/_/  /_/   |___//_/  /_/
                                      \n")
print("Written by Nik Počuča and Michael Gallaugher \n\n")
print("Note genereating test data requires LaplacesDemon installation in R \n")


module MVMix

    import LinearAlgebra, Distances, StatsBase, Statistics, Random, DataStructures, ProgressMeter

    #export mkmeans

    # import functions
    include("MbiClust/mbiclust.jl")
    include("MVNGen/mvngen.jl")
    include("MVNGen/mix_gen.jl")

end # module
