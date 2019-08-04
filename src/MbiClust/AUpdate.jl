using Random, LinearAlgebra, DataStructures

function A_update_init(z,X,M,LoadB,aB,bB,Tol_Sing,N,G,n,q)
    Anew=zeros(n,q,G)
    for g=1:G
        Term1=zeros(n,q)
        Term2=zeros(q,q)
        Mtemp=M[:,:,g]
        aBtemp=aB[:,:,:,g]
        bBtemp=bB[:,:,:,g]
        LoadBtemp=LoadB[:,:,g]
        for i=1:N
            Term1=Term1+z[i,g]*(X[:,:,i]-Mtemp)*LoadBtemp*aBtemp[:,:,i]'
            Term2=Term2+z[i,g]*bBtemp[:,:,i]
        end
        rcond=cond(Term2,1)
        if rcond<Tol_Sing
            return [1]
        end
        Term2inv=inv(Term2)
        Anew[:,:,g]=Term1*Term2inv
    end
    return [0,Anew]
end
