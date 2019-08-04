# Generate Unbiased estimates of M, U , and V from a matrix variate model 

using DataStructures 


function mvunbias(X,mdl)

	# Extract Ms, Us, Vs, etc. 
	Ms = mdl[:M]
	Us = mdl[:U]
	Vs = mdl[:V]

	N = size(X,3)

	result = OrderedDict{Symbol,Any}();

	UsUnb = mapslices(x -> x*(N/(N-1)),Us, dims = [1,2]);	

	result[:Us] = UsUnb;
	result[:Ms] = M;
	result[:Vs] = V;
	return(result) 
end

