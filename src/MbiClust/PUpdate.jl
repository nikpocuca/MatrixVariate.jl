using Random, LinearAlgebra

function Psiup_init(z,X,M,B,LoadA,aB,bB,n,p,N,G,Tol_Sing)
    Psiinv=zeros(p,p,G)
    Psit=zeros(p,p,G)
    Psidet=zeros(G)

    for g=1:G
        denom=sum(z[:,g])*n
        Term1=zeros(p,p)
        Term2=zeros(p,p)
        for i=1:N
            Term1=Term1+z[i,g]*(X[:,:,i]-M[:,:,g])'*LoadA[:,:,g]*(X[:,:,i]-M[:,:,g])
            Term2=Term2+z[i,g]*B[:,:,g]*aB[:,:,i,g]'*LoadA[:,:,g]*(X[:,:,i]-M[:,:,g])
        end
        Psitemp=Term1-Term2
        #print(Diagonal(Psitemp))
        Psi=Diagonal(Psitemp)/denom
        #print(Psi)
        if minimum(diag(Psi))<Tol_Sing
            throw("Problem with Psi Update")
        end
        Psit[:,:,g]=Psi
        Psiinv[:,:,g]=inv(Psi)
        Psidet[g]=logdet(Psiinv[:,:,g])
    end
    return [0,Psit,Psiinv,Psidet]
end
