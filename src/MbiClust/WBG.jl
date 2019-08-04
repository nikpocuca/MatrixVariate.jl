using Random, LinearAlgebra

function WBG_update_init(B,r,Psiinv,G,Tol_Sing)
    iden=Matrix(I,r,r)
    WBG=zeros(r,r,G)
    for g=1:G
        WBGtemp=iden+B[:,:,g]'*Psiinv[:,:,g]*B[:,:,g]
        rcond=1/cond(WBGtemp,1)
        if rcond<Tol_Sing
            return [1]
        end
        WBG[:,:,g]=inv(WBGtemp)
    end
    return [0,WBG]
end
