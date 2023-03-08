import numpy as np
import pandas as pd
from cuml import LinearRegression
from cuml.linear_model import LinearRegression
import cuml

lr = LinearRegression(fit_intercept = True, nprmalize = False, algorithm = "eig")
X = pd.DataFrame()
X['col1'] = np.array([1,1,2,2], dtype=np.float32)
X['col2'] = np.array([1,2,2,3], dtype=np.float32)
y = pd.Series(np.array([6.0, 8.0, 9.0, 11.0], dtype=np.float32))
reg = lr.fit(X,y)

print(reg.coef_)
print(reg.intercept_)

preds = reg.predict(X)
print("lr: ", preds)


lasso = cuml.Lasso()
lasso.fit(X,y)
preds = lasso.predict(X)

print("lasso: ",preds)


elastic = cuml.ElasticNet()
elastic.fit(X,y)
preds = elastic.predict(X)

print("elastic: ", preds)


ridge = cuml.Ridge()
ridge.fit(X,y)
preds = ridge.predict(X)

print("ridge: ", preds)

log = cuml.LogisticRegression()
log.fit(X,y)
preds = log.predict(X)

print("log: ", preds)


gdf_float = pd.DataFrame()
gdf_float['0'] = np.asarray([1.0,2.0,5.0], dtype = np.float32)
gdf_float['1'] = np.asarray([4.0,2.0,1.0], dtype = np.float32)
gdf_float['2'] = np.asarray([4.0,2.0,1.0], dtype = np.float32)

pca_float = cuml.PCA(n_components = 2)
pca_float.fit(gdf_float)

print(f'PCA: {pca_float.components_}')

tsvd_float = cuml.TruncatedSVD(n_components = 2, algorithm = "arpack", n_iter = 20, tol = 1e-9)
tsvd_float.fit(gdf_float)

print(f'tsvd: {tsvd_float.components_}')

hdbscan = cuml.HDBSCAN()
hdbscan.fit(X)

print("hdbscan: ", hdbscan.labels_)

umap = cuml.UMAP()
umap.fit(X)

preds = umap.transform(X)

print("umap: ", preds)
