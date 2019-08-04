using Random, LinearAlgebra

function Estep2_init(M,Sigmainv,A,X,LoadB,G,p,N,q)
    WAG=WAG_update_init(A,q,Sigmainv,G,1e-9)
    if WAG[1]==1
        return [1]
    end
    WAG=WAG[2]
    aB=zeros(q,p,N,G)
    bB=zeros(q,q,N,G)

    for g=1:G
        WAGtemp=WAG[:,:,g]
        Atemp=A[:,:,g]
        Sigmatemp=Sigmainv[:,:,g]
        Mtemp=M[:,:,g]
        for i=1:N
            aB[:,:,i,g]=WAGtemp*Atemp'*Sigmatemp*(X[:,:,i]-Mtemp)
            bB[:,:,i,g]=p*WAGtemp+aB[:,:,i,g]*LoadB[:,:,g]*aB[:,:,i,g]'
        end
    end
    return [0,aB,bB]
end

function Estep3_init(M,Psiinv,B,X,LoadA,G,n,N,r)
    WBG=WBG_update_init(B,r,Psiinv,G,1e-9)
    if WBG[1]==1
        return [1]
    end
    WBG=WBG[2]
    aA=zeros(n,r,N,G)
    bA=zeros(r,r,N,G)

    for g=1:G
        WBGtemp=WBG[:,:,g]
        Btemp=B[:,:,g]
        Psitemp=Psiinv[:,:,g]
        Mtemp=M[:,:,g]
        for i=1:N
            aA[:,:,i,g]=(X[:,:,i]-Mtemp)*Psitemp*Btemp*WBGtemp
            bA[:,:,i,g]=n*WBGtemp+aA[:,:,i,g]'*LoadA[:,:,g]*aA[:,:,i,g]
        end
    end
    return [0,aA,bA]
end
