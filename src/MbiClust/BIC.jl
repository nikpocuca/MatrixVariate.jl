# Calculates the BIC value of the matrix variate model.
using Random, LinearAlgebra

function BICcalc(ll,n,p,q,r,G,N)
    free=(G-1)+G*(n*p+n*q+n-0.5*q*(q-1)+p*r+p-0.5*r*(r-1))
    BIC=2*maximum(ll)-free*log(N)
    return BIC
end
