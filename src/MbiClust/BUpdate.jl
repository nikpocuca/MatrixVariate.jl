using Random, LinearAlgebra

function B_update_init(z,X,M,LoadA,aB,bB,Tol_Sing,N,G,p,r)
    Bnew=zeros(p,r,G)
    for g=1:G
        Term1=zeros(p,r)
        Term2=zeros(r,r)
        Mtemp=M[:,:,g]
        aBtemp=aB[:,:,:,g]
        bBtemp=bB[:,:,:,g]
        LoadAtemp=LoadA[:,:,g]
        for i=1:N
            Term1=Term1+z[i,g]*(X[:,:,i]-Mtemp)'*LoadAtemp*aBtemp[:,:,i]
            Term2=Term2+z[i,g]*bBtemp[:,:,i]
        end
        rcond=cond(Term2,1)
        if rcond<Tol_Sing
            return [1]
        end
        Term2=inv(Term2)
        Bnew[:,:,g]=Term1*Term2
    end
    return [0,Bnew]
end
