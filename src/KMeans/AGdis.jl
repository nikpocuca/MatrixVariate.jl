# Calculates the aggregate within cluster similarity of a kmeans classifcation.
# Based on the paper of Covariate Selection using Telemetric Data by Mario Wuthrich 2018

using Distances

function AGDis(data,labels,centroids)

        # set up Dissimalirty of each cluster
        diss_cluster = zeros(size(centroids,3));

        # iterate through all labels
        for i = 1:size(data,3)

                current_label = Int(labels[i]);
                k_centroid = centroids[:,:,current_label];
                x = data[:,:,i]

                # add the dissimalirty to the respective cluster
                diss_cluster[current_label] += sqeuclidean(x,k_centroid);

        end

        # return total dissimalirty and respective dissimalirty of each cluster
        return sum(diss_cluster), diss_cluster
end
