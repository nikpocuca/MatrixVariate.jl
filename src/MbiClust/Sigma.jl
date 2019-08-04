using Random, LinearAlgebra

function Sigmaup_init(z,X,M,A,LoadB,aB,bB,n,p,N,G,Tol_Sing)
    Sigmat=zeros(n,n,G)
    Sigmainv=zeros(n,n,G)
    Sigmadet=zeros(G)
    for g=1:G
        denom=sum(z[:,g])*p
        Term1=zeros(n,n)
        Term2=zeros(n,n)
        for i=1:N
            Term1=Term1+z[i,g]*(X[:,:,i]-M[:,:,g])*LoadB[:,:,g]*(X[:,:,i]-M[:,:,g])'
            Term2=Term2+z[i,g]*A[:,:,g]*aB[:,:,i,g]*LoadB[:,:,g]*(X[:,:,i]-M[:,:,g])'
        end
        Sigmatemp=(Term1-Term2)/denom
        Sigma=Diagonal(Sigmatemp)
        if minimum(diag(Sigma))<Tol_Sing
            #print(z)
            throw("Tolerance Failure")
            #return[1]
        end
        Sigmat[:,:,g]=Sigma
        # Woodbury
        Sigmainv[:,:,g]=inv(Sigma)
        Sigmadet[g]=logdet(Sigmainv[:,:,g])
    end
    return [0,Sigmat,Sigmainv,Sigmadet]
end
