# Updates mean of Matrix Variate model
using Random, LinearAlgebra

function Mupdate_init(X,z,G,n,p,N)
    M=zeros(n,p,G)
    for g=1:G
        denom=sum(z[:,g])
        numer=zeros(n,p)
        for i=1:N
            numer=numer+z[i,g]*X[:,:,i]
        end
        M[:,:,g]=numer/denom
    end
    return M
end
