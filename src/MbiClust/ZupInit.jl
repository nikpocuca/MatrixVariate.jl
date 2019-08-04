# Updates priori value.
using Random, LinearAlgebra

function zupdate_init(X,pig,M,LoadA,LoadB,detA,detB,n,p,N,G)
    #print(LoadA)
    zmat=zeros(N,G)
    zmat2=zeros(N,G)
    diff=zeros(N)
    #print(N)
    for g=1:G
        for i=1:N
            zmat[i,g]=log(pig[g])+matnormpdf(X[:,:,i],M[:,:,g],LoadA[:,:,g],LoadB[:,:,g],detA[g],detB[g],n,p)
            if isnan(zmat[i,g])
                return [1]
            end
        end
    end

    #

    for i=1:N
        if maximum(zmat[i,:]) < -700
            difer= -700 - maximum(zmat[i,:])
            zmat2[i,:]=zmat[i,:] .+ difer
            diff[i]=difer
        else
            zmat2[i,:] = zmat[i,:]
        end
    end

    zmat2=exp.(zmat2)
    return [zmat2,diff,zmat]
end

function zup_init(zdens,N,G)
    z=zeros(N,G)

    for i in 1:N
            z[i,:]=zdens[i,:]/sum(zdens[i,:])
    end
    return z
end
