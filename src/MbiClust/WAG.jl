using Random, LinearAlgebra

function WAG_update_init(A,q,Sigmainv,G,Tol_Sing)
    prob=0
    iden=Matrix(I,q,q)
    WAG=zeros(q,q,G)
    for g=1:G
        WAGtemp=iden+A[:,:,g]'*Sigmainv[:,:,g]*A[:,:,g]
        rcond=1/cond(WAGtemp,1)
        if rcond<Tol_Sing
            return [1]
        end
        WAG[:,:,g]=inv(WAGtemp)
    end
    return [0,WAG]
end
