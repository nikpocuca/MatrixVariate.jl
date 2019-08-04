# Calculates the log liklihood of the model
using Random, LinearAlgebra

function likcalc(zdens,N,class,z)
    likeclass=0
    for i=1:N
        like=0
        if class[i]==0
            like=log(sum(zdens[1][i,:]))-zdens[2][i]
        else
            like=sum(z[i,:].*zdens[3][i,:])
        end
        likeclass=likeclass+like
    end
    return likeclass
end
