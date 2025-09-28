from sklearn.cluster import KMeans
import numpy as np

def find_clusters(num_colors, num_classes):
    """
    After the KMeans, we have a 256x256 matrix where each cell has
    the id to the parent cluster.
    """
    num_items = 1000
    color_array = np.random.choice(range(256), 3 * num_items).reshape(-1, 3)
    num_classes = num_classes
    label_model = KMeans(n_clusters = num_classes)
    label_model.fit(color_array)
    return 