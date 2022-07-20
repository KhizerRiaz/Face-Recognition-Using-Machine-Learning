# 40 classes
# 400 samples total
# 64x64(image pixels) 4096 dimensionality (features)

# 40 subjects and each subject has 10 diff pictures i.e,400 images

from sklearn.datasets import fetch_olivetti_faces

olivetti_faces = fetch_olivetti_faces()

features = olivetti_faces.data
target = olivetti_faces.target

print(features.shape, "\n", target.shape)
print(target)
# features => 400 rows , 4096 columns i.e, features
# target => 400 rows , 1 column
