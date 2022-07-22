import clustering as clus
from matplotlib import pyplot as plt
import numpy as np



photo_names = []

people = [3, 4, 5, 6] # ids given to each person in 'people' folder
num_photos = [5, 8, 4, 4] # no of photos per person in 'people' folder
for i in people:
    for j in range(1, num_photos[i - 3]):
        photo = "people/p" + str(i) + "_" + str(j) + ".jpg"
        photo_names.append(photo)

print (photo_names)
nodes, adj, dists = clus.create_nodes(photo_names)