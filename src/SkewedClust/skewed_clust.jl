

using SpecialFunctions, Roots, LinearAlgebra, Random, RCall

# dist is either
# "ST"= Skewed T
# "VG" = Variance Gamma
# "NIG" = Normal Inverse Gaussian
# "GH" = Generalized Hyperbolic

function skewedclust(X,G,q,r,dist,initialization,maxiter,seedno,class=nothing,starts=10)
    #Initialize Memberships
    Random.seed!(seedno)
    #dA=Normal(0,1)
    N=size(X)[3]
    n=size(X)[1]
    p=size(X)[2]
    if class==nothing
        class=zeros(N)
    end
    class=convert(Array{Int,1},class)
    #print(class)
    #initstart=F
    init=0
    if initialization=="Gaussian"
        initall=Any[]
        likinit=zeros(starts)
        for s=1:starts
            print(s)
            inittemp=EM_Main_init(X,q,r,G,s,20,class)
            push!(initall,inittemp)
            if inittemp[1]==1
                likinit[s]=-Inf
            else
                likinit[s]=maximum(inittemp[5])
            end
        end
        if (maximum(likinit)==-Inf)
            print("Could Not Initialize")
            return[1]
        end
        init=initall[argmax(likinit)]
        tol=sign(maximum(likinit))*maximum(likinit)/10^4
        #print(init)
        print("initialized")
        #print(init)
        zinit=init[3]
        #zinit=mymap(zinit)
        piginit=init[2]
        #print(piginit)
        Minit=init[6]
        Ainit=zeros(n,p,G)
        Sigmastarinit=init[7]
        Sigmainit=init[9]
        Psistarinit=init[8]
        Psiinit=init[10]
        Loada=init[13]
        Loadb=init[14]
        detSig=init[11]
        detPsi=init[12]
        for g=1:G
            Ainit[:,:,g]=Ainit[:,:,g].+0.1
        end
        pig=piginit
        M=Minit
        #print(Minit)
        A=Ainit
        Sigmastar=Sigmastarinit
        Sigma=Sigmainit
        Psistar=Psistarinit
        Psi=Psiinit
        if dist=="ST"
            Thetaold=[zeros(G).+200.0,0,0]
            #print(Thetaold)
        elseif dist=="GH"
            Thetaold=[zeros(G).+10,zeros(G).+10,0]
        elseif dist=="VG"
            Thetaold=[zeros(G).+50,0,0]
        elseif dist=="NIG"
            Thetaold=[zeros(G).+10,0,0]
        end
        #print(Thetaold)
        #ztemp=Estep1a(X,pig,M,A,Sigmastar,Psistar,detSig,detPsi,Thetaold,dist)
        #znew=ztemp[3]
        znew=zinit
    else
        zinit=zeros(N,G)
        for i=1:N
            temp=rand(1:100,G)
            zinit[i,:]=temp/sum(temp)
        end
        for i=1:N
            if class[i]>0
                temp=zeros(G)
                temp[class[i]]=1
                zinit[i,:]=temp
            end
        end
        #print(zinit)
        #zinit=mymap(zinit)
        piginit=pigupdate(zinit,G,N)
        #print(piginit)
        Minit=zeros(n,p,G)
        Ainit=zeros(n,p,G)
        Sigmastarinit=zeros(n,n,G)
        Sigmainit=zeros(n,n,G)
        Psistarinit=zeros(p,p,G)
        Psiinit=zeros(p,p,G)
        Loada=zeros(n,q,G)
        Loadb=zeros(p,r,G)
        detSig=zeros(G)
        detPsi=zeros(G)
        for g=1:G
            for i=1:N
                Minit[:,:,g]=Minit[:,:,g]+zinit[i,g]*X[:,:,i]/sum(zinit[:,g])
            end
            Sigmatemp=zeros(n,n)
            Psitemp=zeros(p,p)
            for i=1:N
                Sigmatemp=Sigmatemp+(zinit[i,g]*(X[:,:,i]-Minit[:,:,g])*(X[:,:,i]-Minit[:,:,g])'/(sum(zinit[:,g])*p))
                Psitemp=Psitemp+(zinit[i,g]*(X[:,:,i]-Minit[:,:,g])'*(X[:,:,i]-Minit[:,:,g])/(n*sum(zinit[:,g])))
            end
            #print(1/cond(Sigmatemp))
            Sigmainit[:,:,g]=inv(Diagonal(Sigmatemp))
            Ainit[:,:,g]=Ainit[:,:,g].+0.1
            Psiinit[:,:,g]=inv(Diagonal(Psitemp))
            Loada[:,:,g]=reshape(rand(-1:0.1:1,n*q),n,q)
            Loadb[:,:,g]=reshape(rand(-1:0.1:1,p*r),p,r)
            Sigmastarinit[:,:,g]=inv(inv(Sigmainit[:,:,g])+Loada[:,:,g]*Loada[:,:,g]')
            #print(1/cond(Sigmastarinit[:,:,g]))
            Psistarinit[:,:,g]=inv(inv(Psiinit[:,:,g])+Loadb[:,:,g]*Loadb[:,:,g]')
            detSig[g]=logdet(Sigmastarinit[:,:,g])
            detPsi[g]=logdet(Psistarinit[:,:,g])
        end
        pig=piginit
        M=Minit
        A=Ainit
        #print(Minit)
        Sigmastar=Sigmastarinit
        Sigma=Sigmainit
        Psistar=Psistarinit
        Psi=Psiinit
        if dist=="ST"
            Thetaold=[zeros(G).+200.0,0,0]
            #print(Thetaold)
        elseif dist=="GH"
            Thetaold=[zeros(G).+10,zeros(G).+10,0]
        elseif dist=="VG"
            Thetaold=[zeros(G).+200,0,0]
        elseif dist=="NIG"
            Thetaold=[zeros(G).+10,0,0]
        end
        znew=zinit
    end
    #print(znew)
    Estep1=Estep1b(X,M,A,Sigmastar,Psistar,Thetaold,dist)
    a=Estep1[1]
    b=Estep1[2]
    c=Estep1[3]
    loglik=zeros(maxiter)
    lik=zeros(maxiter)
    conv=0
    iter=1
    #print(tol)
    flagM=0
    Mold=M
    ztemp=0
    while conv==0
        #print(iter)
        CM1eval=CM1(X,znew,G,N,a,b,c,Thetaold,dist,flagM,Mold)
        if CM1eval[1]==1
            print("Problem with first maximization step")
            return [1]
        end

        pig=CM1eval[2]
        #print(pig)
        if flagM==0
            Mold=M
            M=CM1eval[3]
        else
            M=Mold
        end
        A=CM1eval[4]
        Thetaold=CM1eval[5]
        #print(Thetaold)
        #print(M[:,:,1])
        #print(M)
        #print(Thetaold)
        #ztemp=Estep1a(X,pig,M,A,Sigmastar,Psistar,detSig,detPsi,Thetaold,dist)
        #znew=ztemp[3]
        #diff=ztemp[4]
        #print(sum(diff))
        #dens=ztemp[2]
        #liktemp=0
        #for i=1:N

        #    liktemp=liktemp+log(sum(dens[i,:]))
        #end
        #loglik[iter]=liktemp-sum(diff)
        #lik[iter]=liktemp
        #if iter>1
        #    if (loglik[iter]-loglik[iter-1])<0
        #        print("Decreasing Likelihood after 1st estep Q=$q R=$r G=$G")
        #        print(loglik[iter]-loglik[iter-1])
        #        return [pig,M,A,Sigma,Psi,Thetaold,Loada,Loadb,Sigmastar,Psistar,znew,loglik[1:iter]]
        #    end
        #end
        #Estep1=Estep1b(X,M,A,Sigmastar,Psistar,Thetaold,dist)
        #a=Estep1[1]
        #b=Estep1[2]
        #c=Estep1[3]
        #iter=iter+1
        #print(iter)
        CM2=AECM2(X,q,znew,a,b,M,A,Loada,Sigma,Psistar)
        if CM2[1]==1
            print("Problem with second maximization step")
            return [1]
        end
        Sigma=CM2[2]
        Sigmastar=CM2[3]
        detSig=CM2[4]
        #print(detSig)
        Loada=CM2[5]
        CM3=AECM3(X,r,znew,a,b,M,A,Loadb,Sigmastar,Psi)
        if CM3[1]==1
            print("Problem with third maximization step")
            return [1]
        end
        Psi=CM3[2]
        Psistar=CM3[3]
        detPsi=CM3[4]
        Loadb=CM3[5]
        #print(Loadb)
        ztemp=Estep1a(X,pig,M,A,Sigmastar,Psistar,detSig,detPsi,Thetaold,dist)
        if ztemp[1]==1
            print("NaN in zig updates")
            #print(pig)
            return [1]
        end
        if flagM==0
            if ztemp[1]==2
                print("M Set")
                flagM=1
                M=Mold
                CM1eval=CM1(X,znew,G,N,a,b,c,Thetaold,dist,flagM,M)
                if CM1eval[1]==1
                    print("Problem with first maximization step")
                    return [1]
                end
                A=CM1eval[4]
                Thetaold=CM1eval[5]
                CM2=AECM2(X,q,znew,a,b,M,A,Loada,Sigma,Psistar)
                if CM2[1]==1
                    print("Problem with second maximization step")
                    return [1]
                end
                Sigma=CM2[2]
                Sigmastar=CM2[3]
                detSig=CM2[4]
                #print(detSig)
                Loada=CM2[5]
                CM3=AECM3(X,r,znew,a,b,M,A,Loadb,Sigmastar,Psi)
                if CM3[1]==1
                    print("Problem with third maximization step")
                    return [1]
                end
                Psi=CM3[2]
                Psistar=CM3[3]
                detPsi=CM3[4]
                Loadb=CM3[5]
                #print(Loadb)
                ztemp=Estep1a(X,pig,M,A,Sigmastar,Psistar,detSig,detPsi,Thetaold,dist)
                if ztemp[1]==1
                    print("NaN in zig updates")
                    #print(pig)
                    return [1]
                end
                if ztemp[1]==2
                    print("Infinite updates")
                    #print(pig)
                    return [1]
                end
            end
        end
        if ztemp[1]==2
            print("Infinite updates")
            #print(pig)
            return [1]
        end
        znew=ztemp[3]
        for i=1:N
            if class[i]>0
                temp=zeros(G)
                temp[class[i]]=1
                znew[i,:]=temp
            end
        end
        #print(znew)
        diff=ztemp[4]
        #print("This is Thursday")
        #print(sum(diff))
        dens=ztemp[2]
        logdens=ztemp[1]
        #print(dens)
        classpred=mymap2(znew)

        likclass=0
        for i=1:N
            liktemp=0
            if class[i]==0
                liktemp=log(sum(dens[i,:]))-diff[i]
            else
                liktemp=sum(znew[i,:].*logdens[i,:])
            end
            likclass=likclass+liktemp
        end
        liktemp=likclass
        if initialization!="Gaussian"
            if iter==5
                tol=sign(liktemp)*liktemp/10^3
            end
        end
        loglik[iter]=likclass
        #print(loglik[iter])
        lik[iter]=likclass
        #print(liktemp)

        if iter>1
            if (loglik[iter]-loglik[iter-1])<0
                print("Decreasing Likelihood Q=$q R=$r G=$G")
                print([loglik[iter]-loglik[iter-1]])
                BIC=-Inf
                return [1,pig,M,A,Sigma,Psi,Thetaold,Loada,Loadb,Sigmastar,Psistar,znew,classpred,loglik[1:iter],BIC]
            end
        end

        if (iter>5)
            #print([tol])
            if loglik[iter-1]-loglik[iter-2]==0
                BIC=BICcalc(loglik[1:iter],n,p,q,r,G,N,dist)
                return [0,pig,M,A,Sigma,Psi,Thetaold,Loada,Loadb,Sigmastar,Psistar,znew,classpred,loglik[1:iter],BIC]
            end

            ak=(lik[iter]-lik[iter-1])/(lik[iter-1]-lik[iter-2])
            linf=lik[iter-1]+(lik[iter]-lik[iter-1])/(1-ak)

            if (abs(linf-lik[iter-1]))<tol
                print("converged")
                print([G,q,r])
                BIC=BICcalc(loglik[1:iter],n,p,q,r,G,N,dist)
                return [0,pig,M,A,Sigma,Psi,Thetaold,Loada,Loadb,Sigmastar,Psistar,znew,classpred,loglik[1:iter],dist,BIC,class,[q,r]]
            end
        end

        if iter==maxiter
            print("Maximum Iterations Reached")
            BIC=BICcalc(loglik[1:iter],n,p,q,r,G,N,dist)
            return [1,pig,M,A,Sigma,Psi,Thetaold,Loada,Loadb,Sigmastar,Psistar,znew,classpred,loglik[1:iter]]
        end

        Estep1=Estep1b(X,M,A,Sigmastar,Psistar,Thetaold,dist)
        a=Estep1[1]
        b=Estep1[2]
        c=Estep1[3]
        iter=iter+1

    end
end





function pigupdate(z,G,N)
    pig=zeros(G)
    for g=1:G
        pig[g]=sum(z[:,g])/N
    end
    return pig
end

function Mupdate(X,a,b,z,Ng)
    n=size(X)[1]
    p=size(X)[2]
    N=size(X)[3]
    G=size(z)[2]
    M=zeros(n,p,G)
    az=a.*z
    bz=b.*z
    for g=1:G
        abarg=sum(az[:,g])/Ng[g]
        bsum=sum(bz[:,g])
        denom=abarg*bsum-Ng[g]
        numer=zeros(n,p)
        for i=1:N
            numer=numer+z[i,g]*X[:,:,i]*(abarg*b[i,g]-1)
        end
        M[:,:,g]=numer/denom
    end
    return M
end

function Aupdate(X,a,b,z,Ng,flagM,M=nothing)
    n=size(X)[1]
    #print(b)
    p=size(X)[2]
    N=size(X)[3]
    G=size(z)[2]
    A=zeros(n,p,G)
    az=a.*z
    bz=b.*z
    if flagM==0
        for g=1:G
            abarg=sum(az[:,g])/Ng[g]
            bbarg=sum(bz[:,g])/Ng[g]
            bsum=sum(bz[:,g])
            denom=abarg*bsum-Ng[g]
            numer=zeros(n,p)
            for i=1:N
                numer=numer+z[i,g]*X[:,:,i]*(bbarg-b[i,g])
            end
            A[:,:,g]=numer/denom
        end
    else
        for g=1:G
            asum=sum(az[:,g])
            numer=zeros(n,p)
            for i=1:N
                numer=numer+z[i,g]*(X[:,:,i]-M[:,:,g])
            end
            A[:,:,g]=numer/asum
        end
    end
    return A
end

function CM1(X,z,G,N,a,b,c,Thetaold,dist,flagM,Mold=nothing)
    Ng=zeros(G)
    for g=1:G
        Ng[g]=sum(z[:,g])
    end
    #print(Ng)
        pignew=pigupdate(z,G,N)
        Mnew=Mupdate(X,a,b,z,Ng)
        Anew=Aupdate(X,a,b,z,Ng,flagM,Mold)
        Thetanew=Thetaup(a,b,c,z,G,Thetaold,dist)
        if Thetanew[3]==1
            return [1]
        end



    return [0,pignew,Mnew,Anew,Thetanew]
end


function AECM2(X,q,z,a,b,M,A,Loada,Sigma,Psistar)

    Esteptemp=Estep2(q,X,a,b,M,A,Sigma,Psistar,Loada)
    if Esteptemp[1]==1
        return [1]
    end
        Loadanewtemp=Loadaup(q,z,X,M,A,Psistar,Esteptemp)
        if Loadanewtemp[1]==1
            return [1]
        end
        Loadanew=Loadanewtemp[2]
        Sigmatemp=Sigmaup(X,z,a,b,M,A,Psistar,Loadanew,Esteptemp)
        if Sigmatemp[1]==1
            return [1]
        end
        Sigmainvnew=Sigmatemp[2]
        Sigmastarnew=Sigmatemp[3]
        detSignew=Sigmatemp[4]

        return [0,Sigmainvnew,Sigmastarnew,detSignew,Loadanew]

end
function Lgupdate(q,Loada,Sigma)
    G=size(Sigma)[3]
    n=size(Sigma)[1]
    #print(Sigma)
    Lg=zeros(q,n,G)
    coef=zeros(q,q,G)
    for g=1:G
        coeftemp=Matrix(I,q,q)+Loada[:,:,g]'*Sigma[:,:,g]*Loada[:,:,g]
        rcond=1/cond(coeftemp)
        if rcond<1e-6
            return[1]
        end
        coef[:,:,g]=inv(coeftemp)
        #print(coef[:,:,g])
        Lg[:,:,g]=coef[:,:,g]*Loada[:,:,g]'*Sigma[:,:,g]
    end
    E2=[0,coef,Lg]
    return E2
end

function Estep2(q,X,a,b,M,A,Sigma,Psistar,Loada)
    p=size(X)[2]
    N=size(X)[3]
    G=size(Psistar)[3]
    E12=zeros(q,p,N,G)
    E22=zeros(q,p,N,G)
    E32=zeros(q,q,N,G)
    L=Lgupdate(q,Loada,Sigma)
    if L[1]==1
        return [1]
    end
    Lg=L[3]
    coef=L[2]
    for g=1:G
        Mg=M[:,:,g]
        Ag=A[:,:,g]
        Ltemp=Lg[:,:,g]
        coeftemp=coef[:,:,g]
        Psig=Psistar[:,:,g]
        for i=1:N
            Xtemp=X[:,:,i]
            E12[:,:,i,g]=Ltemp*(Xtemp-Mg-a[i,g]*Ag)
            E22[:,:,i,g]=Ltemp*(b[i,g]*(Xtemp-Mg)-Ag)
            E32[:,:,i,g]=p*coeftemp+b[i,g]*Ltemp*(Xtemp-Mg)*Psig*(Xtemp-Mg)'Ltemp'-Ltemp*((Xtemp-Mg)*Psig*Ag'+Ag*Psig*(Xtemp-Mg)')*Ltemp'+a[i,g]*Ltemp*Ag*Psig*Ag'Ltemp'
        end
    end

    E=[0,E12,E22,E32]
    return E
end

function Loadaup(q,z,X,M,A,Psistar,E)
    N=size(X)[3]
    G=size(M)[3]
    n=size(M)[1]
    Loada=zeros(n,q,G)
    for g=1:G
        temp1=zeros(n,q)
        temp2=zeros(q,q)
        Mg=M[:,:,g]
        Ag=A[:,:,g]
        E1=E[2][:,:,:,g]
        E2=E[3][:,:,:,g]
        E3=E[4][:,:,:,g]
        Psig=Psistar[:,:,g]
        for i=1:N
            Xtemp=X[:,:,i]
            temp1=temp1+z[i,g]*((Xtemp-Mg)*Psig*E2[:,:,i]'-Ag*Psig*E1[:,:,i]')
            temp2=temp2+z[i,g]*E3[:,:,i]
        end
        rcond=1/cond(temp2)
        if rcond<1e-6
            return [1]
        end
        Loada[:,:,g]=temp1*inv(temp2)
    end
    return [0,Loada]
end

function Sigmaup(X,z,a,b,M,A,Psistar,Loada,E)
    p=size(X)[2]
    n=size(X)[1]
    G=size(z)[2]
    N=size(X)[3]
    Sigmanew=zeros(n,n,G)
    Sigmastarnew=zeros(n,n,G)
    Sigmanewinv=zeros(n,n,G)
    detSigmastar=zeros(G)
    for g=1:G
        Ng=sum(z[:,g])
        Psig=Psistar[:,:,g]
        E1=E[2][:,:,:,g]
        E2=E[3][:,:,:,g]
        E3=E[4][:,:,:,g]
        Mg=M[:,:,g]
        Ag=A[:,:,g]
        Loadg=Loada[:,:,g]
        Temp=zeros(n,n)
        for i=1:N
            Xtemp=X[:,:,i]
            Temp=Temp+z[i,g]*(b[i,g]*(Xtemp-Mg)*Psig*(Xtemp-Mg)'-(Ag+Loadg*E2[:,:,i])*Psig*(Xtemp-Mg)'-(Xtemp-Mg)*Psig*(Ag+Loadg*E2[:,:,i])'+a[i,g]*Ag*Psig*Ag'+Loadg*E1[:,:,i]*Psig*Ag'+Ag*Psig*E1[:,:,i]'*Loadg'+Loadg*E3[:,:,i]*Loadg')
        end
        Sigmatemp=Temp/(Ng*p)
        Sigmanew[:,:,g]=Diagonal(Sigmatemp)
        if minimum(diag(Sigmanew[:,:,g]))<1e-6
            return [1]
        end
        #print(Sigmanew)
        Sigmanewinv[:,:,g]=inv(Sigmanew[:,:,g])
        Sigmastartemp=Sigmanew[:,:,g]+Loadg*Loadg'
        rcond=1/cond(Sigmastartemp)
        if rcond<1e-6
            return [1]
        end
        Sigmastarnew[:,:,g]=inv(Sigmastartemp)
        detSigmastar[g]=logdet(Sigmastarnew[:,:,g])
    end
    #print(Sigmanew)
    return [0,Sigmanewinv,Sigmastarnew,detSigmastar]
end


function AECM3(X,r,z,a,b,M,A,Loadb,Sigmastar,Psi)

    Esteptemp=Estep3(r,X,a,b,M,A,Sigmastar,Psi,Loadb)
    if Esteptemp[1]==1
        return [1]
    end
    #print(Esteptemp)
        Loadbnewtemp=Loadbup(r,z,X,M,A,Sigmastar,Esteptemp)
        if Loadbnewtemp[1]==1
            return [1]
        end
        Loadbnew=Loadbnewtemp[2]
        Psitemp=Psiup(X,z,a,b,M,A,Sigmastar,Loadbnew,Esteptemp)
        if Psitemp[1]==1
            return [1]
        end
        Psiinvnew=Psitemp[2]
        Psistarnew=Psitemp[3]
        detPsinew=Psitemp[4]
        #print(detPsinew)

        return [0,Psiinvnew,Psistarnew,detPsinew,Loadbnew]

end
function Dgupdate(r,Loadb,Psi)
    G=size(Psi)[3]
    p=size(Psi)[1]
    Dg=zeros(p,r,G)
    coef=zeros(r,r,G)
    for g=1:G
        coeftemp=Matrix(I,r,r)+Loadb[:,:,g]'*Psi[:,:,g]*Loadb[:,:,g]
        rcond=1/cond(coeftemp)
        if rcond<1e-6
            return[1]
        end
        coef[:,:,g]=inv(coeftemp)
        #print(coef[:,:,g])
        Dg[:,:,g]=Psi[:,:,g]*Loadb[:,:,g]*coef[:,:,g]
    end
    E2=[0,coef,Dg]
    return E2
end
function Estep3(r,X,a,b,M,A,Sigmastar,Psi,Loadb)
    n=size(X)[1]
    N=size(X)[3]
    G=size(Sigmastar)[3]
    #print(G)
    E13=zeros(n,r,N,G)
    E23=zeros(n,r,N,G)
    E33=zeros(r,r,N,G)
    D=Dgupdate(r,Loadb,Psi)
    Dg=D[3]
    coef=D[2]
    if D[1]==1
        return[1]
    end
    for g=1:G
        Mg=M[:,:,g]
        Ag=A[:,:,g]
        Dtemp=Dg[:,:,g]

        coeftemp=coef[:,:,g]
        Sigg=Sigmastar[:,:,g]
        for i=1:N
            Xtemp=X[:,:,i]
            E13[:,:,i,g]=(Xtemp-Mg-a[i,g]*Ag)*Dtemp
            E23[:,:,i,g]=(b[i,g]*(Xtemp-Mg)-Ag)*Dtemp
            E33[:,:,i,g]=n*coeftemp+b[i,g]*Dtemp'*(Xtemp-Mg)'*Sigg*(Xtemp-Mg)*Dtemp-Dtemp'*((Xtemp-Mg)'*Sigg*Ag+Ag'*Sigg*(Xtemp-Mg))*Dtemp+a[i,g]*Dtemp'*Ag'*Sigg*Ag*Dtemp
        end
    end
    E=[0,E13,E23,E33]
    return E
end



function Loadbup(r,z,X,M,A,Sigstar,E)
    N=size(X)[3]
    G=size(M)[3]
    p=size(M)[2]
    Loadb=zeros(p,r,G)
    for g=1:G
        temp1=zeros(p,r)
        temp2=zeros(r,r)
        Mg=M[:,:,g]
        Ag=A[:,:,g]
        E1=E[2][:,:,:,g]
        E2=E[3][:,:,:,g]
        E3=E[4][:,:,:,g]
        Sigg=Sigstar[:,:,g]
        for i=1:N
            Xtemp=X[:,:,i]
            temp1=temp1+z[i,g]*((Xtemp-Mg)'*Sigg*E2[:,:,i]-Ag'*Sigg*E1[:,:,i])
            temp2=temp2+z[i,g]*E3[:,:,i]
        end
        rcond=1/cond(temp2)
        if rcond<1e-6
            return [1]
        end
        Loadb[:,:,g]=temp1*inv(temp2)
    end
    return [0,Loadb]
end




function Psiup(X,z,a,b,M,A,Sigstar,Loadb,E)
    p=size(X)[2]
    n=size(X)[1]
    G=size(z)[2]
    N=size(X)[3]
    Psinew=zeros(p,p,G)
    Psistarnew=zeros(p,p,G)
    Psinewinv=zeros(p,p,G)
    detPsistar=zeros(G)
    for g=1:G
        Ng=sum(z[:,g])
        #print(Ng)
        Sigg=Sigstar[:,:,g]
        E1=E[2][:,:,:,g]
        E2=E[3][:,:,:,g]
        E3=E[4][:,:,:,g]
        Mg=M[:,:,g]
        Ag=A[:,:,g]
        Loadg=Loadb[:,:,g]
        Temp=zeros(p,p)
        for i=1:N
            Xtemp=X[:,:,i]
            Temp=Temp+z[i,g]*(b[i,g]*(Xtemp-Mg)'*Sigg*(Xtemp-Mg)-(Ag'+Loadg*E2[:,:,i]')*Sigg*(Xtemp-Mg)-(Xtemp-Mg)'*Sigg*(Ag+E2[:,:,i]*Loadg')+a[i,g]*Ag'*Sigg*Ag+Loadg*E1[:,:,i]'*Sigg*Ag+Ag'*Sigg*E1[:,:,i]*Loadg'+Loadg*E3[:,:,i]*Loadg')
        end
        Psitemp=Temp/(Ng*n)
        Psinew[:,:,g]=Diagonal(Psitemp)
        if minimum(diag(Psinew[:,:,g]))<1e-6
            return [1]
        end
        Psinewinv[:,:,g]=inv(Psinew[:,:,g])
        Psistartemp=Psinew[:,:,g]+Loadg*Loadg'
        rcond=1/cond(Psistartemp)
        if rcond<1e-6
            return [1]
        end
        Psistarnew[:,:,g]=inv(Psistartemp)
        detPsistar[g]=logdet(Psistarnew[:,:,g])
    end
    return [0,Psinewinv,Psistarnew,detPsistar]
end

function mybessel(nu,z)
    bes=try besselk(nu,z)
    catch
        bes=nothing
    end
    if bes==nothing
        bes2=0.5*(log(pi)-log(2)-log(nu))-nu*log(exp(1)*z)+nu*log(2*nu)
        return bes2
        #print("approx")
    elseif bes==0
        bes3=try besselkx(nu,z)
        catch
            bes3=nothing
        end
        if bes3==nothing
            bes4=0.5*(log(pi)-log(2)-log(nu))-nu*log(exp(1)*z)+nu*log(2*nu)
            return bes4
        elseif bes3<1e-300
            bes5=log(1e-300)
            return(bes5)
        else
            return log(bes3)-z
        end
        #print("check")
    else
        bes=log(bes)
    end
    return bes
end

function bessderiv(nu,z)
    eps=0.001
    #test=try besselkx(abs(nu),z)
    #catch
    #    test=nothing
    #end
    #if test==nothing
    #    derivtype="approxexact"
    #elseif test==0
    #    derivtype="approxexact"
    #else
    #    derivtype="approx"
    #end

    #if derivtype=="approx"
    bessdev=(mybessel(abs(nu+eps),z)-mybessel(abs(nu),z))/eps
    #else
    #    bessdev=-(1/(2*abs(nu)))-log(z)+log(2*abs(nu))
    #end
    return bessdev
end

function Egig(nu,a,b,func="x")
    z=sqrt(a*b)
    #print([z,nu])
    if func=="x"
        temp=0.5*(log(b)-log(a))+(mybessel(abs(nu+1),z)-mybessel(abs(nu),z))
        expec=exp(temp)
    elseif func=="1/x"
        expec=exp(0.5*(log(a)-log(b))+(mybessel(abs(nu+1),z)-mybessel(abs(nu),z)))-2*nu/b
    elseif func=="logx"
        expec=0.5*(log(b)-log(a))+bessderiv(nu,z)
    end

    return expec
end

function Estep1b(X,M,A,Sigmastar,Psistar,Theta,dist)
    G=size(M)[3]
    #print(G)
    N=size(X)[3]
    n=size(X)[1]
    p=size(X)[2]
    ai=zeros(N,G)
    bi=zeros(N,G)
    ci=zeros(N,G)
    if dist=="ST"
        for g=1:G
            rho=tr(Sigmastar[:,:,g]*A[:,:,g]*Psistar[:,:,g]*A[:,:,g]')
            #print(rho)
            nug=Theta[1][g]
            lambtemp=-(nug+n*p)/2
            for i=1:N
                delta=tr(Sigmastar[:,:,g]*(X[:,:,i]-M[:,:,g])*Psistar[:,:,g]*(X[:,:,i]-M[:,:,g])')
                #print((delta+Theta[1][g]))
                ai[i,g]=Egig(lambtemp,rho,(delta+Theta[1][g]),"x")
                bi[i,g]=Egig(lambtemp,rho,(delta+Theta[1][g]),"1/x")
                ci[i,g]=Egig(lambtemp,rho,(delta+Theta[1][g]),"logx")
            end
        end
        res=[ai,bi,ci]
        return res
    elseif dist=="GH"
        for g=1:G
            rho=tr(Sigmastar[:,:,g]*A[:,:,g]*Psistar[:,:,g]*A[:,:,g]')
            for i=1:N
                delta=tr(Sigmastar[:,:,g]*(X[:,:,i]-M[:,:,g])*Psistar[:,:,g]*(X[:,:,i]-M[:,:,g])')
                lambtemp=Theta[2][g]-n*p/2
                ai[i,g]=Egig(lambtemp,rho+Theta[1][g],delta+Theta[1][g],"x")
                bi[i,g]=Egig(lambtemp,rho+Theta[1][g],delta+Theta[1][g],"1/x")
                ci[i,g]=Egig(lambtemp,rho+Theta[1][g],delta+Theta[1][g],"logx")
            end
            res=[ai,bi,ci]
        end
        return res
    elseif dist=="VG"
        for g=1:G
            rho=tr(Sigmastar[:,:,g]*A[:,:,g]*Psistar[:,:,g]*A[:,:,g]')
            for i=1:N
                delta=tr(Sigmastar[:,:,g]*(X[:,:,i]-M[:,:,g])*Psistar[:,:,g]*(X[:,:,i]-M[:,:,g])')
                lambtemp=Theta[1][g]-n*p/2
                ai[i,g]=Egig(lambtemp,rho+2*Theta[1][g],delta,"x")
                bi[i,g]=Egig(lambtemp,rho+2*Theta[1][g],delta,"1/x")
                ci[i,g]=Egig(lambtemp,rho+2*Theta[1][g],delta,"logx")
            end
            res=[ai,bi,ci]
        end
        return res
    elseif dist=="NIG"
        for g=1:G
            gammt=Theta[1][g]
            rho=tr(Sigmastar[:,:,g]*A[:,:,g]*Psistar[:,:,g]*A[:,:,g]')
            for i=1:N
                delta=tr(Sigmastar[:,:,g]*(X[:,:,i]-M[:,:,g])*Psistar[:,:,g]*(X[:,:,i]-M[:,:,g])')
                lambtemp=-(1+n*p)/2
                ai[i,g]=Egig(lambtemp,(rho+gammt^2),(delta+1),"x")
                bi[i,g]=Egig(lambtemp,(rho+gammt^2),(delta+1),"1/x")
            end
            res=[ai,bi,ci]

        end
        return res
    end
end

function Estep1a(X,pig,M,A,Sigma,Psi,detSig,detPsi,Theta,dist)
    G=size(M)[3]
    #print(Theta)
    #print(Psi)
    Tol=-700
    N=size(X)[3]
    logden=zeros(N,G)
    z=zeros(N,G)
    dens=zeros(N,G)
    delta=zeros(N,G)
    n=size(X)[1]
    p=size(X)[2]
    #print(A)
    for g=1:G
        #print(detSig[g])
        #print(detPsi[g])
        if (dist=="GH")
            Thetag=[Theta[1][g],Theta[2][g]]
        else
            Thetag=Theta[1][g]
        end
        #print(Thetag)
        Mg=M[:,:,g]
        Ag=A[:,:,g]
        Sigmag=Sigma[:,:,g]
        Psig=Psi[:,:,g]
        detSigg=detSig[g]
        detPsig=detPsi[g]
        #print(Mg)
        #print(Thetag)
        for i=1:N
            Xtemp=X[:,:,i]
            logden[i,g]=log(pig[g])+logdens(Xtemp,Mg,Ag,Sigmag,Psig,Thetag,detSigg,detPsig,dist)
            if isnan(logden[i,g])
                return [1]
            end
        end
    end
    #print(logden)
    #print(delta)
    diff=zeros(N)
    for i=1:N
        if maximum(logden[i,:])< Tol
            diff[i]=-700-maximum(logden[i,:])
        end
        dens[i,:]=exp.(logden[i,:].+diff[i])
        if isinf(maximum(dens[i,:]))
            return [2]
        end
        z[i,:]=dens[i,:]/sum(dens[i,:])
    end

    return [logden,dens,z,diff]
end

function logdens(X,M,A,Sigma,Psi,Theta,detSig,detPsi,dist)
    n=size(X)[1]
    p=size(X)[2]
    if dist=="ST"
        nu=Theta
        #print(nu)
        rho=tr(Sigma*A*Psi*A')
        delta=tr(Sigma*(X-M)*Psi*(X-M)')
        tau=-(nu+n*p)/2

        logdens=mybessel(abs(tau),sqrt((delta+nu)*rho))+log(2)+0.5*nu*log(nu/2)+tr(Sigma*(X-M)*Psi*A')-(n*p/2)*log(2*pi)+(n/2)*detPsi+(p/2)*detSig-log(gamma(nu/2))+(tau/2)*(log(delta+nu)-log(rho))

        return logdens
    elseif dist=="GH"
        omega=Theta[1]
        lambda=Theta[2]
        rho=tr(Sigma*A*Psi*A')
        delta=tr(Sigma*(X-M)*Psi*(X-M)')
        tau=lambda-(n*p/2)
        logdens=tr(Sigma*(X-M)*Psi*A')-(n*p/2)*log(2*pi)+(n/2)*detPsi+(p/2)*detSig-mybessel(abs(lambda),omega)+(tau/2)*(log(delta+omega)-log(rho+omega))+mybessel(abs(tau),sqrt((delta+omega)*(rho+omega)))
        return logdens
    elseif dist=="VG"
        gamm=Theta
        rho=tr(Sigma*A*Psi*A')
        delta=tr(Sigma*(X-M)*Psi*(X-M)')
        tau=gamm-(n*p/2)
        logdens=log(2)+gamm*log(gamm)+tr(Sigma*(X-M)*Psi*A')-(n*p/2)*log(2*pi)+(n/2)*detPsi+(p/2)*detSig-log(gamma(gamm))+(tau/2)*(log(delta)-log(rho+2*gamm))+mybessel(abs(tau),sqrt((delta)*(rho+2*gamm)))
        return logdens
    elseif dist=="NIG"
        gammt=Theta
        rho=tr(Sigma*A*Psi*A')
        delta=tr(Sigma*(X-M)*Psi*(X-M)')
        tau=-(1+n*p)/2
        logdens=log(2)+gammt+tr(Sigma*(X-M)*Psi*A')-((n*p+1)/2)*log(2*pi)+(n/2)*detPsi+(p/2)*detSig+(tau/2)*(log(delta+1)-log(rho+gammt^2))+mybessel(abs(tau),sqrt((delta+1)*(rho+gammt^2)))
        return logdens
    end
end

function Rfunc(lam,om)
    R=exp(mybessel(abs(lam+1),om)-mybessel(abs(lam),om))
    return R
end

function Thetaup(a,b,c,z,G,Thetaold,dist)
    N=size(z)[1]
    #print(z)
    if dist=="ST"
        Theta=[zeros(G),0,0]
        for g=1:G
            #print(g)
            sumzbc=0
            for i=1:N
                sumzbc=sumzbc+z[i,g]*(b[i,g]+c[i,g])
            end
            #print(sumzbc)
            #print(sumzbc)
            #print((1/(sum(z[:,g])))*sumzbc)
            STf(x)=log(x/2)+1-digamma(x/2)-(1/(sum(z[:,g])))*sumzbc
            nutest= try find_zero(STf,(1e-30,1e70))
            catch
                nutest=nothing
            end
            #show(nutest)
            #print(nutest)

            if nutest==nothing
                res=[0,0,1]
                return res
            elseif nutest>200
                Theta[1][g]=200
            elseif nutest<2
                Theta[1][g]=2
            else
                Theta[1][g]=nutest
            end
        end
    elseif dist=="VG"
        Theta=[zeros(G),0,0]
        for g=1:G
            sumzbc=0
            Ng=sum(z[:,g])
            abarg=0
            cbarg=0
            for i=1:N
                abarg=abarg+z[i,g]*a[i,g]/Ng
                cbarg=cbarg+z[i,g]*c[i,g]/Ng
            end
            f(x)=log(x)+1-digamma(x)+cbarg-abarg
            nutest=try find_zero(f,(1e-3,2000))
            catch
                nutest=nothing
            end

            if nutest==nothing
                res=[0,0,1]
                return res
            elseif nutest>100
                Theta[1][g]=100
            elseif nutest<1
                Theta[1][g]=1
            else
                Theta[1][g]=nutest
            end
        end
    elseif dist=="NIG"
        Theta=[zeros(G),0,0]
        for g=1:G
            abarg=0
            Ng=sum(z[:,g])
            for i=1:N
                abarg=abarg+a[i,g]*z[i,g]
            end
            Theta[1][g]=Ng/abarg
        end
    elseif dist=="GH"
        Theta=[zeros(G),zeros(G),0]
        lamold=Thetaold[2]
        omold=Thetaold[1]
        for g=1:G
            cbarg=0
            abarg=0
            bbarg=0
            Ng=sum(z[:,g])
            for i=1:N
                cbarg=cbarg+c[i,g]*z[i,g]/Ng
                abarg=abarg+a[i,g]*z[i,g]/Ng
                bbarg=bbarg+b[i,g]*z[i,g]/Ng
            end
            denom=bessderiv(lamold[g],omold[g])
            lamnew=cbarg*lamold[g]/denom

            firstder=0.5*(Rfunc(lamnew,omold[g])+Rfunc((-lamnew),omold[g])-(abarg+bbarg))
            secondder=0.5*((Rfunc(lamnew,omold[g]))^2-((1+2*lamnew)/omold[g])*Rfunc(lamnew,omold[g])-1+(Rfunc(-lamnew,omold[g]))^2-((1-2*lamnew)/omold[g])*Rfunc((-lamnew),omold[g])-1)
            omnew=omold[g]-firstder/secondder

            if omnew<1
                omnew=1
            end
            Theta[2][g]=lamnew
            Theta[1][g]=omnew
        end
    end
    return Theta
end


function BICcalc(ll,n,p,q,r,G,N,dist)
    if (dist=="GH")
        free=(G-1)+G*(2*n*p+n*q+n-0.5*q*(q-1)+p*r+p-0.5*r*(r-1))+2*G
    else
        free=(G-1)+G*(2*n*p+n*q+n-0.5*q*(q-1)+p*r+p-0.5*r*(r-1))+G
    end
    BIC=2*maximum(ll)-free*log(N)
    return BIC
end

function mymap(z)
    znew=zeros(size(z)[1],size(z)[2])
    N=size(z)[1]
    for i=1:N
        znew[i,indmax(z[i,:])]=1
    end
    return znew
end


#test=EMMain(dat2,QTR[30,1],QTR[30,2],QTR[30,3],"VG","Gaussi",1000,30)
#test=EMMain(datfinal,2,QTR[16,1],QTR[16,2],"VG","Other",4000,16,classknown)
#function SkewedMVFA(X,G=1:5,Q=1:5,R=1:5,inititers=50,EMiters=4000)





#cd("/Users/michaelgallaugher/Desktop/PhD/MVF_Skew/Julia_Code2/")
