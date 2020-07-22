from fcmeans import FCM
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from seaborn import scatterplot as scatter

n_samples = 50000
n_bins    = 3
centers   = [(-5,-5), (0,0), (5,5)]

X,_ = make_blobs(n_samples=n_samples, n_features = 2, cluster_std = 1.0, centers = centers, shuffle = False, random_state=42)

fcm = FCM(n_clusters = 3)
fcm.fit(X)

fcm_centers = fcm.centers
fcm_labels  = fcm.u.argmax(axis = 1)

print(fcm_centers)


f, axes = plt.subplots(1, 2, figsize=(11,5))
scatter(X[:,0], X[:,1], ax=axes[0])
scatter(X[:,0], X[:,1], ax=axes[1], hue=fcm_labels)
scatter(fcm_centers[:,0], fcm_centers[:,1], ax=axes[1],marker="s",s=200)
plt.show()
