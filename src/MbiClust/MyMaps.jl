# Map functions used to assign memberships of posteriors.
using Random, LinearAlgebra

function mymap(z,N)
    class=zeros(N)
    for i=1:N
        class[i]=indmax(z[i,:])
    end
    return class
end

function mymap2(z)
    N=size(z)[1]
    class=zeros(N)
    for i=1:N
        class[i]=argmax(z[i,:])
    end
    return class
end
