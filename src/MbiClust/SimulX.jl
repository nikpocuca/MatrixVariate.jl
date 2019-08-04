using Random, LinearAlgebra
# IDK WHAT THIS DOES.

function Simulx(M,Sigma,Psi,n,p,G,NG)
    x=matnormal(NG[1],n,p,M[:,:,1],Sigma[:,:,1],Psi[:,:,1])
    for g=2:G
        x=cat(3,x,matnormal(NG[g],n,p,M[:,:,g],Sigma[:,:,g],Psi[:,:,g]))
    end
    return x
end
