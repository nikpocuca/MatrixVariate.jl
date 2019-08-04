#=
 Function: mbiclust (clusters matrix variate data and performs factor analysis)
 model: Mixture of Matrix Variate Bilinear Factor Analyzers (https://arxiv.org/pdf/1712.08664.pdf)

 Arguments:
 X (row -r by column c by N sized multidimensional array double)
 q,r (row and column factors)
 G (number of groups)
 maxiter (maximum number of EM iterations allowed)
 class (1 by N vector denoting class memberships, 0 means no class and will run unsupervised)
 seedno (seed number of random variable setting)
 Tol_Sigma (Positive definite tolerance for covariance matrices)

=#

using ProgressMeter


include("PiUpdate.jl")
include("LikeCalc.jl")
include("MyMaps.jl")
include("BIC.jl")
include("MatrixNormalLogPdf.jl")
include("ZupInit.jl")
include("MeanUpdate.jl")
include("SimulX.jl")
include("WAG.jl")
include("WBG.jl")
include("EStep.jl")
include("AUpdate.jl")
include("BUpdate.jl")
include("Sigma.jl")
include("PUpdate.jl")
include("AECM.jl")


function mbiclust(X,q,r,; G = 1, maxiter = 2000, class = zeros(size(X,3)), seedno = 10, Tol_Sigma = 1e-6)

    # Run Kmeans
    init_mems = convert(Array{Int,1},mkmeans(X,k = G)[2])
    k_init_mems = init_mems

    class=convert(Array{Int,1},class)
    N=size(X)[3]; n=size(X)[1]; p=size(X)[2];

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

    # Update Probability observation i belongs to group g
    piginit=pigupdate_init(zinit,G,N)

    # Generate zero vectors and matrices for all parameters
    Minit=zeros(n,p,G)
    Sigmastarinit=zeros(n,n,G); Sigmainit=zeros(n,n,G);
    Psistarinit=zeros(p,p,G); Psiinit=zeros(p,p,G);
    Loada=zeros(n,q,G); Loadb=zeros(p,r,G);
    detSig=zeros(G); detPsi=zeros(G);

    # Begin Iteration for generating initial parameters
    for g=1:G # For g in 1 - G groups
        for i=1:N # For i in 1:N observation
            # Update the Mean matrix for the gth group
            Minit[:,:,g] += zinit[i,g]*X[:,:,i]/sum(zinit[:,g])
        end

        # Generate a temporary Row CoVar and Col CoVar
        Sigmatemp=zeros(n,n)
        Psitemp=zeros(p,p)

        for i=1:N # Iterate through all observations and generate Row CoVar and Col CoVar
            Sigmatemp=Sigmatemp+(zinit[i,g]*(X[:,:,i]-Minit[:,:,g])*(X[:,:,i]-Minit[:,:,g])'/(sum(zinit[:,g])*p))
            Psitemp=Psitemp+(zinit[i,g]*(X[:,:,i]-Minit[:,:,g])'*(X[:,:,i]-Minit[:,:,g])/(n*sum(zinit[:,g])))
        end

        # Invert diagonals and calculate loads
        Sigmainit[:,:,g]=inv(Diagonal(Sigmatemp))
        Psiinit[:,:,g]=inv(Diagonal(Psitemp))
        Loada[:,:,g]=reshape(rand(-1:0.1:1,n*q),n,q)
        Loadb[:,:,g]=reshape(rand(-1:0.1:1,p*r),p,r)

        # Sudo Sigma with loads
        Sinit_g = Sigmainit[:,:,g];
        Loada_g = Loada[:,:,g];

        Sigmastarinit[:,:,g]=Sinit_g-Sinit_g*Loada_g*inv(Matrix(I,q,q)+Loada_g'*Sinit_g*Loada_g)*Loada_g'*Sinit_g
        detSig[g]=logdet(Matrix(I,q,q)-Loada_g'*Sigmastarinit[:,:,g]*Loada_g)-log(prod(diag(Sigmatemp)))

        Psistarinit[:,:,g]=Psiinit[:,:,g]-Psiinit[:,:,g]*Loadb[:,:,g]*inv(Matrix(I,r,r)+Loadb[:,:,g]'*Psiinit[:,:,g]*Loadb[:,:,g])*Loadb[:,:,g]'*Psiinit[:,:,g]

        InnerTerm = Matrix(I,r,r)-Loadb[:,:,g]'*Psistarinit[:,:,g]*Loadb[:,:,g]
        Left_term = logdet(InnerTerm);
        detPsi[g]=Left_term -log(prod(diag(Psitemp)))

    end

    # Set up iteratiable variables.
    # These variables will change in each EM iteration.
    pig=piginit
    M=Minit
    Sigmastar=Sigmastarinit; Sigma=Sigmainit;
    Psistar=Psistarinit; Psi=Psiinit;
    z=zinit

    # Set acceleration criteria and iteration counts
    conv=0; tol=0; lik=zeros(maxiter); iter=1

    pr = ProgressUnknown("Progress out of Max Iterations")

    for i=1:maxiter # Begin EM Iterations

        # =================== AECM Stage 1 ================================== #

	ProgressMeter.next!(pr; showvalues = [(:iter,i)])

	ECM1=AECM1_init(z,X,G,n,p,N); pig=ECM1[1];
        if pig[1]==NaN
            throw("AECM1_init Failure: Stage 1 Failed")
        end

        # Use Results from Stage 1
        M=ECM1[2]

        # =================== AECM Stage 2 ================================== #

        ECM2=AECM2_init(z,X,M,Loada,Sigma,Psistar,N,n,p,q,G,Tol_Sigma)

        if ECM2[1]==1
            throw("AECM2_init Failure: Stage 2 Failed")
        end

        # Use Results from Stage 2
        Loada=ECM2[2]; Sigma=ECM2[3];
        detSig=ECM2[4]; Sigmastar=ECM2[5];

        # =================== AECM Stage 3 ================================== #

        ECM3=AECM3_init(z,X,M,Loadb,Psi,Sigmastar,N,n,p,r,G,Tol_Sigma)
        if ECM3[1]==1
            throw("AECM3_init Failure: Stage 3 Failed")
        end

        # Use Results from Stage 3
        Loadb=ECM3[2]; Psi=ECM3[3];
        detPsi=ECM3[4]; Psistar=ECM3[5];

        # ================== Stage Set ====================================== #

        # Begin Update of zdens - Posteriori
        zdens=zupdate_init(X,pig,M,Sigmastar,Psistar,detSig,detPsi,n,p,N,G)

        if zdens[1]==1
            throw("zupdate_init Failure")
        end

        z=zup_init(zdens[1],N,G)

        # Calculate class prediction using soft classification to hard.
        classpred=mymap2(z)


        # Calculate Liklihood
        # Aikieke Accleration criterion, check convergence and return parameters

        lik[i]=likcalc(zdens,N,class,z)
        if iter==5
            # Tolerance for decreasing liklihood, needs to be changed for general
            tol=sign(lik[i])*lik[i]/10^4
        end


        if iter>1
            if (lik[iter]-lik[iter-1])<0
                likeDiff = lik[iter]-lik[iter-1]
                print("Decreasing Likelihood: LikeDiff=$likeDiff, Q=$q, R=$r, G=$G. \n")
                return [1,pig,z,iter,lik[1:maxiter],M,Sigmastar,Psistar]
            end
        end

        if (iter>5)
            if lik[iter-1]-lik[iter-2]==0
                print("Liklihood difference is 0 \n\n")
                BIC=BICcalc(lik[1:iter],n,p,q,r,G,N)
                returnVar = OrderedDict{Symbol,Any}();
                algoCondition = "Lik Diff 0"
                returnVar[:algoCondition] = algoCondition;
                returnVar[:pig] = pig; returnVar[:M] = M;
                returnVar[:Sigma] = Sigma; returnVar[:Psi] = Psi;
                returnVar[:Loada] = Loada; returnVar[:Loadb] = Loadb;
                returnVar[:Sigmastar] = Sigmastar; returnVar[:Psistar] = Psistar;
                returnVar[:z] = z; returnVar[:classpred] = classpred;
                returnVar[:likliehoods] = lik[1:iter]; returnVar[:BIC] = BIC;
                returnVar[:classpred] = classpred; returnVar[:class] = class;
                returnVar[:q] = q; returnVar[:r] = r;

                return returnVar
            end

            ak=(lik[iter]-lik[iter-1])/(lik[iter-1]-lik[iter-2])
            linf=lik[iter-1]+(lik[iter]-lik[iter-1])/(1-ak)

            if (abs(linf-lik[iter-1]))<tol
                print("\n Converged\n\n"); BIC=BICcalc(lik[1:iter],n,p,q,r,G,N);
                returnVar = OrderedDict{Symbol,Any}();
                algoCondition = "Converged"
                returnVar[:algoCondition] = algoCondition;
                returnVar[:pig] = pig; returnVar[:M] = M;
                returnVar[:Sigma] = Sigma; returnVar[:Psi] = Psi;
                returnVar[:Loada] = Loada; returnVar[:Loadb] = Loadb;
                returnVar[:Sigmastar] = Sigmastar; returnVar[:Psistar] = Psistar;
                returnVar[:z] = z; returnVar[:classpred] = classpred;
                returnVar[:likliehoods] = lik[1:iter]; returnVar[:BIC] = BIC;
                returnVar[:classpred] = classpred; returnVar[:class] = class;
                returnVar[:q] = q; returnVar[:r] = r;
                returnVar[:k_init_mems] = k_init_mems;

                return returnVar
            end
        end
        iter += 1
    end

    print("Maximum Iterations Reached \n\n")
    returnVar = OrderedDict{Symbol,Any}();
    algoCondition = "Max Iterations Reached"
    returnVar[:algoCondition] = algoCondition; returnVar[:pig] = pig;
    returnVar[:z] = z; returnVar[:iter] = iter;
    returnVar[:likliehoods] = lik[1:maxiter];
    returnVar[:M] = M; returnVar[:Sigmastar] = Sigmastar;
    returnVar[:Psistar] = Psistar; returnVar[:Sigma] = Sigma;
    returnVar[:Psi] = Psi; returnVar[:detSig] = detSig;
    returnVar[:detPsi] = detPsi; returnVar[:Loada]; returnVar[:Loadb]

    return returnVar
end
