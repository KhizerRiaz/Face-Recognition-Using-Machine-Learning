# 40 classes
# 400 samples total
# 64x64(image pixels) 4096 dimensionality (features)

# 40 subjects and each subject has 10 diff pictures i.e,400 images

from sklearn.datasets import fetch_olivetti_faces
from matplotlib import pyplot as plt
import numpy as np

olivetti_faces = fetch_olivetti_faces()

features = olivetti_faces.data
targets = olivetti_faces.target

fig, sub_plots = plt.subplots(nrows=5, ncols=8, figsize=(14, 8))
sub_plots = sub_plots.flatten()  # converted to 1D array
# print(sub_plots)

for unique_user_id in np.unique(targets):
    image_index = unique_user_id * 10  # (0-9 for first) ....
    # print(image_index)
    sub_plots[unique_user_id].imshow(features[image_index].reshape(64, 64), cmap="gray")
    # sub_plots[unique_user_id].set_xticks([])
    # sub_plots[unique_user_id].set_yticks([])
    sub_plots[unique_user_id].set_title("FACE : %s" % unique_user_id)


plt.show()
# for index, value in enumerate(targets):
#     print(index, value)
