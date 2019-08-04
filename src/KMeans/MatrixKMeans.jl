# Matrix-K-Means Module for clustering matrix variate data
# Must be a square matrix
# n x n x N Matrix

using StatsBase, Distances, Statistics


function mkmeans(data; k = 1 , max_iter = 100, threshold = 0.001, metric = sqeuclidean)
        # Check dimensions of data
        checkData(data)

        # Pick Centroids for start
        s_index =  sample(1:size(data,3), k, replace = false)
        centroids = data[:,:, s_index]

        # Create a copy
        n_centroids = copy(centroids)

        # start an empty array for our cluster ids. This will hold the cluster assignment
        cluster_ids = zeros(size(data,3))
        # Keep track of the number of iterations
        for i in 1:max_iter
              # Iterate over each point
              for col_idx in 1:size(data, 3)
                  # let's index the ponts one by one
                  p = data[:,:,col_idx]

                  # calculate the distance between the point and each centroids
                  distances = mapslices( x -> sqeuclidean(p,x), centroids,dims = [1,2])
                  # updated for Julia <1.0 distances = mapslices(x -> calcMatrixDistance(p,x,metric) , centroids, [1,2])
                  # we calculate the squared Euclidian distance

                  # now find the index of the closest centroid

                  cluster_ids[col_idx] = findmin(distances)[2][3]
                  # this gives the index of the minimum
              end

            # Iterate over each centroid
            for cluster_id in 1:size(centroids, 3)
                # find the mean of the assigned points for that particluar cluster
                n_centroids[:,:,cluster_id] = mapslices(mean, data[:,:, cluster_id .== cluster_ids], dims = 3)
            end

            # now measure the total distance that the centroids moved
            center_change = sqeuclidean(centroids,n_centroids)

            centroids = copy(n_centroids)

            # if the centroids move negligably, then we're done
            if center_change < threshold
                # println(i)
                break
            end

        end

        return centroids, cluster_ids
    end

    function checkData(D)
        sizeD = size(D)
        if (length(sizeD) != 3)
            throw("Please check dimensions of entire data, should be n x n x N")

        elseif ( sizeD[1] != sizeD[2] || sizeD[1] > sizeD[3] || sizeD[2] > sizeD[3])
            throw("Please check size of matrix, should be a square matrix with n x n x N")
        end
end

# Calculates the diference between matrix and centroid
# For now calculates the squared eucledian distance but later I will
# add a function where I will pass in different distance
function calcMatrixDistance(X_n,c,f_metric)
    return(f_metric(X_n,c))
end

#=
# Testing Dataset



function generateTestSet()
    R"library(LaplacesDemon)"
    R" X <- matrix(0,2,2)"
    R"
    for(i in 1:100){
    X <- cbind(X,rmatrixnorm(matrix(c(1,2,1,4),2,2),matrix(c(1,0,0,1),2,2),matrix(c(1,0,0,1),2,2)) )}
    X_2 <- matrix(0,2,2)
    for(i in 1:100){
    X_2 <- cbind(X_2,rmatrixnorm(matrix(c(1,2,2,1),2,2),matrix(c(1,0,0,1),2,2),matrix(c(1,0,0,1),2,2)) )}
    "
    R"X <- cbind(X,X_2)"
    R"X <- array(X,c(2,2,200))"
    @rget X
end



function generateTestSet2()
    R"library(LaplacesDemon)"
    R" X <- matrix(0,2,2)"
    R"
    for(i in 1:100){
    X <- cbind(X,rmatrixnorm(matrix(c(1,2,1,4),2,2),matrix(c(1,0,0,1),2,2),matrix(c(1,0,0,1),2,2)) )}
    X_2 <- matrix(0,2,2)
    for(i in 1:100){
    X_2 <- cbind(X_2,rmatrixnorm(matrix(c(1,2,2,1),2,2),matrix(c(1,0,0,1),2,2),matrix(c(1,0,0,1),2,2)) )}
    "
    R"X_3 <- matrix(0,2,2)"
    R"for(i in 1:100){
        X_3 <- cbind(X_3,rmatrixnorm(matrix(c(6,1,20,3),2,2),matrix(c(1,0,0,1),2,2),matrix(c(1,0,0,1),2,2)) )}"
    R"X <- cbind(X,X_2,X_3)"
    R"X <- array(X,c(2,2,300))"
    @rget X
end



=#
