using Random, LinearAlgebra

function AECM1_init(z,X,G,n,p,N)
    pig=pigupdate_init(z,G,N)
    M=Mupdate_init(X,z,G,n,p,N)
    return [pig,M]
end

function AECM2_init(z,X,M,A,Sigmainv,LoadB,N,n,p,q,G,Tol_Sing)
    Estep=Estep2_init(M,Sigmainv,A,X,LoadB,G,p,N,q)

    if Estep[1]==1
        #print("Probelm with sedond E Step")
        return [1]
    end

    aB=Estep[2]
    bB=Estep[3]
    Anew=A_update_init(z,X,M,LoadB,aB,bB,Tol_Sing,N,G,n,q)
    if Anew[1]==1
        throw("Problem with A update")
        #return [1]
    end
    Anew=Anew[2]
    Sigmanew=Sigmaup_init(z,X,M,A,LoadB,aB,bB,n,p,N,G,Tol_Sing)
    if Sigmanew[1]==1
        throw("Problem with Sigma Update")
        #return [1]
    end
    LoadA=zeros(n,n,G)
    detA=zeros(G)
    for g=1:G
        LoadAtemp=Sigmanew[2][:,:,g]+Anew[:,:,g]*Anew[:,:,g]'
        rcond=cond(LoadAtemp)
        if rcond<Tol_Sing
            print("Singular LoadA")
            return [1]
        end
        # Woodbury Identity
        LoadA[:,:,g] = Sigmanew[3][:,:,g] - Sigmanew[3][:,:,g]*Anew[:,:,g]*inv(Matrix(I,q,q)+Anew[:,:,g]'*Sigmanew[3][:,:,g]*Anew[:,:,g])*Anew[:,:,g]'*Sigmanew[3][:,:,g]
        detA[g]=logdet(Matrix(I,q,q)-Anew[:,:,g]'*LoadA[:,:,g]*Anew[:,:,g])-log(prod(diag(Sigmanew[2][:,:,g])))

        #LoadA[:,:,g]=inv(LoadAtemp)
        #detA[g]=logdet(LoadA[:,:,g])

    end
    return [0,Anew,Sigmanew[3],detA,LoadA]
end

function AECM3_init(z,X,M,B,Psiinv,LoadA,N,n,p,r,G,Tol_Sing)
    Estep=Estep3_init(M,Psiinv,B,X,LoadA,G,n,N,r)

    if Estep[1]==1
        #print("Probelm with third E Step")
        return [1]
    end

    aB=Estep[2]
    bB=Estep[3]
    Bnew=B_update_init(z,X,M,LoadA,aB,bB,Tol_Sing,N,G,p,r)
    if Bnew[1]==1
        throw("Problem with B update")
    end
    Bnew=Bnew[2]
    Psinew=Psiup_init(z,X,M,B,LoadA,aB,bB,n,p,N,G,Tol_Sing)
    if Psinew[1]==1
        throw("Problem with Psi Update")
    end
    LoadB=zeros(p,p,G)
    detB=zeros(G)
    for g=1:G
        LoadBtemp=Psinew[2][:,:,g]+Bnew[:,:,g]*Bnew[:,:,g]'
        rcond=cond(LoadBtemp)
        if rcond<Tol_Sing
            throw("Singular LoadB")
        end
        #LoadB[:,:,g]=inv(LoadBtemp)
        #detB[g]=logdet(LoadB[:,:,g])
        LoadB[:,:,g]=Psinew[3][:,:,g]-Psinew[3][:,:,g]*Bnew[:,:,g]*inv(Matrix(I,r,r)+Bnew[:,:,g]'*Psinew[3][:,:,g]*Bnew[:,:,g])*Bnew[:,:,g]'*Psinew[3][:,:,g]
        detB[g]=logdet(Matrix(I,r,r)-Bnew[:,:,g]'*LoadB[:,:,g]*Bnew[:,:,g])-log(prod(diag(Psinew[2][:,:,g])))
    end
    return [0,Bnew,Psinew[3],detB,LoadB]
end
