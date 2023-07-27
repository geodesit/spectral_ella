from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from get_data_sets import *

hdr = get_hdr_load()
bands = get_bands()
scalar = StandardScaler()
scaled_data = scalar.fit_transform(bands)
print(scaled_data)
pca = PCA(n_components=10)
pca.fit(scaled_data)
data_pca = pca.transform(scaled_data)
print(data_pca)
