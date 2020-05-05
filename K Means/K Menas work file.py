import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

digits = load_digits()
data = scale(digits.data)
y = data.targets

k = len(np.unique(y))
sample, features = data.shape