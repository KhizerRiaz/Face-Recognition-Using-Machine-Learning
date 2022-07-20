from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

olivetti_faces = fetch_olivetti_faces()

features = olivetti_faces.data
targets = olivetti_faces.target

fig, sub_plots = plt.subplots(nrows=1, ncols=10, figsize=(18, 9))
sub_plots = sub_plots.flatten()

# for 1st person
for i in range(10):
    sub_plots[i].imshow(features[i].reshape(64, 64), cmap="gray")

plt.show()
