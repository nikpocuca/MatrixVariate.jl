# Calculates the probability of X from a matrix normal density using log version
using Random, LinearAlgebra

function matnormpdf(X,M,Sigmastar,Psistar,detSig,detPsi,n,p)
    logdens=-n*p*log(2*pi)/2+(p/2)*(detSig)+(n/2)*(detPsi)-(1/2)*tr(Sigmastar*(X-M)*Psistar*(X-M)')
    return logdens
end



function matnormpdf2(X_i;M_g,U_inv_g,V_inv_g,detU_g,detV_g,r,c)
    logdens=-r*c*log(2*pi)/2-(c/2)*log(detU_g)-(r/2)*log(detV_g)-(1/2)*tr(V_inv_g*(X_i-M_g)*U_inv_g*(X_i-M_g)')
    return logdens
end
