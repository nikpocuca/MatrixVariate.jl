# Calculates the probability of observation i belonging to group g
using Random, LinearAlgebra

function pigupdate_init(z,G,N)
    Pi_g=zeros(G)
    for g=1:G
        Pi_g[g]=sum(z[:,g])/N
    end
    return Pi_g
end
