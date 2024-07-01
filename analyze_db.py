import sqlite3
import itertools
from collections import defaultdict
import roaringbitmap

from util.data_util import *
from util.timer import *

# For K-Means Method
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# DBSCAN
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

def fetch_existing_combinations_as_bitmap(db_path, table_name, columns):
    """
    Fetches existing combinations of column values from the database and stores them in a Roaring Bitmap.
    
    Args:
    db_path (str): Path to the SQLite database file.
    table_name (str): Name of the table to query.
    columns (list): List of column names to query.
    
    Returns:
    roaringbitmap.RoaringBitmap: A bitmap representing existing combinations in the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT id, {', '.join(columns)} FROM {table_name}"
    cursor.execute(query)
    
    bitmap = roaringbitmap.RoaringBitmap()
    for row in cursor.fetchall():
        # Convert the row to a single integer
        bitmap.add(row[0])
    
    conn.close()
    return bitmap

def find_missing_combinations(db_path, table_name, data_util):
    """
    Finds missing combinations of column values in the database table.
    
    Args:
    db_path (str): Path to the SQLite database file.
    table_name (str): Name of the table to query.
    possible_values (dict): A dictionary where keys are column names and values are lists of possible values for those columns.
    
    Returns:
    list: A list of tuples representing missing combinations.
    """
    columns = data_util.columns
    existing_bitmap = fetch_existing_combinations_as_bitmap(db_path, table_name, columns)
    
    missing_combinations = []
    for combo_and_hash in data_util.generate_all_combinations():
        if combo_and_hash[1] not in existing_bitmap:
            missing_combinations.append(combo_and_hash[0])
    
    return missing_combinations

# === Hole Distance ===

# =======================
# Hamming Distance Method
# =======================
def hamming_distance(a, b):
    """Calculate the Hamming distance between two sequences, considering grouped values."""
    distance = 0
    merged_el = list()
    for el1, el2 in zip(a, b):
        if isinstance(el1, tuple) and isinstance(el2, tuple):
            el1_set = set(el1)
            el2_set = set(el2)
            if el1_set == el2_set:
                merged_el.append(tuple(el1_set))
            elif (el1_set <= el2_set) or (el2_set <= el1_set):
                merged_el.append(tuple(el1_set | el2_set))
            elif bool(el1_set & el2_set):
                merged_el.append(tuple(el1_set | el2_set))
                distance += 1
            else:
                distance += 1
        elif isinstance(el1, tuple):
            distance += 1
            merged_el.append(el1)
        elif isinstance(el2, tuple):
            distance += 1
            merged_el.append(el2)
        elif el1 != el2:
            distance += 1
            merged_el.append((el1, el2))
        else: # Match
            merged_el.append(el1)
    if distance != 1:
        merged_el = None
    else:
        merged_el = tuple(merged_el)
    return distance, merged_el

def aggregate_holes(holes, final_holes_set):
    """Aggregate holes with a Hamming distance of 1, using [x, y] notation."""
    new_holes = set()
    for hole1, hole2 in itertools.combinations(holes, 2):
        distance, merged = hamming_distance(hole1, hole2)
        if merged != None: # New hole was created
            new_holes.add(merged)
            final_holes_set.add(merged)
            # Remove 
            try:
                final_holes_set.remove(hole1)
            except:
                pass
            try:
                final_holes_set.remove(hole2)
            except:
                pass
            # print('Hole 1:', hole1, 'Hole 2:', hole2, distance, merged)
        else: # Holes did not yield new holes
            pass

    if len(new_holes) > 1:
        aggregate_holes(new_holes, final_holes_set)

    return

def iterative_aggregation(holes):
    """Iteratively aggregate holes until no more aggregation is possible."""
    condensed_holes = set(holes)
    aggregate_holes(holes, condensed_holes)
    return condensed_holes

def calculate_size(hole):
    """Calculate the size of a hole by multiplying the lengths of its elements."""
    size = 1
    for element in hole:
        if isinstance(element, tuple):
            size *= len(element)
        else:
            size *= 1
    return size

def sort_holes_by_size(holes):
    """Sort the holes by their calculated size."""
    holes_with_size = [(hole, calculate_size(hole)) for hole in holes]
    sorted_holes = sorted(holes_with_size, key=lambda x: x[1])
    return sorted_holes

def print_sorted_holes(sorted_holes):
    """Print the sorted holes with their sizes."""
    for hole, size in sorted_holes:
        print(f"Hole: {hole}, Size: {size}")


# =======================
# K-Means Method
# =======================
def kmeans_plot_elbow_method(data, max_k=10):
    wcss = []
    max_k = min(len(data), max_k)
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, max_k + 1), wcss, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.show()

def kmeans_find_best_k_with_silhouette(data, max_k=10):
    best_k = 2
    best_score = -1
    max_k = min(len(data)-1, max_k)
    for i in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_k = i
    return best_k

def kmeans_preprocess_missing_combinations(missing_combinations):
    """Convert missing combinations to a numpy array."""
    return np.array(missing_combinations)

def kmeans_apply_kmeans_clustering(data, n_clusters):
    """Apply K-means clustering to the data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

def kmeans_summarize_cluster(cluster_data):
    """Summarize a cluster of combinations."""
    summary = []
    for i in range(cluster_data.shape[1]):
        unique_values = np.unique(cluster_data[:, i])
        if len(unique_values) == 1:
            summary.append(unique_values[0])
        else:
            summary.append(list(unique_values))
    return summary

def kmeans_summarize_clusters(data, labels, n_clusters):
    """Generate a summary for each cluster."""
    cluster_summaries = []
    for cluster in range(n_clusters):
        cluster_data = data[labels == cluster]
        summary = summarize_cluster(cluster_data)
        cluster_summaries.append(summary)
    return cluster_summaries

def kmeans_format_cluster_summary(cluster_summaries):
    """Format the cluster summaries into a readable format."""
    formatted_summaries = []
    for i, summary in enumerate(cluster_summaries):
        formatted_summary = []
        for item in summary:
            if isinstance(item, list):
                formatted_summary.append(f"[{', '.join(map(str, item))}]")
            else:
                formatted_summary.append(str(item))
        formatted_summaries.append(f"Cluster {i}: {', '.join(formatted_summary)}")
    return formatted_summaries

def kmeans_summary(holes):
    # Step 1: Preprocess the data
    data = kmeans_preprocess_missing_combinations(holes)

    # Step 2: Find the optimal number of clusters
    kmeans_plot_elbow_method(data)
    best_k = kmeans_find_best_k_with_silhouette(data)
    best_k = 3
    print(f'Optimal number of clusters: {best_k}')

    # Step 3: Apply K-means clustering
    labels, cluster_centers = kmeans_apply_kmeans_clustering(data, best_k)

    # Step 4: Summarize the clusters
    cluster_summaries = kmeans_summarize_clusters(data, labels, best_k)

    # Step 5: Format the summaries for readability
    formatted_summaries = kmeans_format_cluster_summary(cluster_summaries)

    # Print the summaries
    for i, summary in enumerate(cluster_summaries):
        print(f"Cluster {i}: {summary}")

# =======================
# DBSCAN Method
# =======================
def dbscan_hamming_distance_matrix(data):
    """Compute the Hamming distance matrix."""
    return squareform(pdist(data, metric='hamming'))

def dbscan_apply_dbscan_clustering(data, eps, min_samples):
    """Apply DBSCAN clustering with Hamming distance."""
    distance_matrix = dbscan_hamming_distance_matrix(data)
    print(distance_matrix)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = model.fit_predict(distance_matrix)
    print(labels)
    return labels

def dbscan_summarize_cluster(cluster_data):
    """Summarize a cluster of combinations."""
    summary = []
    for i in range(cluster_data.shape[1]):
        unique_values = np.unique(cluster_data[:, i])
        if len(unique_values) == 1:
            summary.append(int(unique_values[0]))  # Convert np.int64 to int
        else:
            summary.append([int(val) for val in unique_values])  # Convert np.int64 to int in list
    return summary

def dbscan_summarize_clusters(data, labels):
    """Generate a summary for each cluster."""
    cluster_summaries = []
    for cluster in np.unique(labels):
        if cluster == -1:  # Skip noise points
            continue
        cluster_data = data[labels == cluster]
        summary = dbscan_summarize_cluster(cluster_data)
        cluster_summaries.append(summary)
    return cluster_summaries

def dbscan_format_cluster_summary(cluster_summaries):
    """Format the cluster summaries into a readable format."""
    formatted_summaries = []
    for i, summary in enumerate(cluster_summaries):
        formatted_summary = []
        for item in summary:
            if isinstance(item, list):
                formatted_summary.append(f"[{', '.join(map(str, item))}]")
            else:
                formatted_summary.append(str(item))
        formatted_summaries.append(f"Cluster {i}: {', '.join(formatted_summary)}")
    return formatted_summaries

def dbscan_summary(holes):
    # Step 1: Preprocess the data
    data = np.array(holes)
    print(data)

    # Step 2: Apply DBSCAN Clustering
    eps = 0.4  # This is the maximum distance between two samples for them to be considered as in the same neighborhood
    min_samples = 2  # This is the number of samples in a neighborhood for a point to be considered as a core point
    labels = dbscan_apply_dbscan_clustering(data, eps, min_samples)

    # Step 3: Summarize the clusters
    cluster_summaries = dbscan_summarize_clusters(data, labels)

    # Step 4: Format the summaries for readability
    formatted_summaries = dbscan_format_cluster_summary(cluster_summaries)

    # Print the formatted summaries
    for summary in formatted_summaries:
        print(summary)

# =======================
# Agglomerative Clustering Method
# =======================
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def ac_hamming_distance_matrix(data):
    """Compute the Hamming distance matrix."""
    return squareform(pdist(data, metric='hamming'))

def ac_apply_agglomerative_clustering_with_threshold(data, distance_threshold):
    """Apply Agglomerative Clustering with Hamming distance and distance threshold."""
    distance_matrix = ac_hamming_distance_matrix(data)
    model = AgglomerativeClustering(
        metric='precomputed', 
        linkage='complete', 
        distance_threshold=distance_threshold, 
        n_clusters=None  # n_clusters is determined by the distance threshold
    )
    labels = model.fit_predict(distance_matrix)
    return labels, model

def ac_summarize_cluster(cluster_data):
    """Summarize a cluster of combinations."""
    summary = []
    for i in range(cluster_data.shape[1]):
        unique_values = np.unique(cluster_data[:, i])
        if len(unique_values) == 1:
            summary.append(int(unique_values[0]))  # Convert np.int64 to int
        else:
            summary.append([int(val) for val in unique_values])  # Convert np.int64 to int in list
    return summary

def ac_summarize_clusters(data, labels):
    """Generate a summary for each cluster."""
    cluster_summaries = []
    unique_labels = np.unique(labels)
    for cluster in unique_labels:
        if cluster == -1:  # Skip noise points
            continue
        cluster_data = data[labels == cluster]
        summary = ac_summarize_cluster(cluster_data)
        cluster_summaries.append(summary)
    return cluster_summaries

def ac_format_cluster_summary(cluster_summaries):
    """Format the cluster summaries into a readable format."""
    formatted_summaries = []
    for i, summary in enumerate(cluster_summaries):
        formatted_summary = []
        for item in summary:
            if isinstance(item, list):
                formatted_summary.append(f"[{', '.join(map(str, item))}]")
            else:
                formatted_summary.append(str(item))
        formatted_summaries.append(f"Cluster {i}: {', '.join(formatted_summary)}")
    return formatted_summaries

def ac_plot_dendrogram(model, **kwargs):
    """Create a dendrogram plot."""
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

def ac_summary(holes, plot=False):
    # Step 1: Preprocess the data
    data = np.array(holes)

    # Step 2: Apply Agglomerative Clustering with Distance Threshold
    distance_threshold = 0.4  # This can be adjusted based on the desired clustering sensitivity
    labels, model = ac_apply_agglomerative_clustering_with_threshold(data, distance_threshold)

    # Step 3a: Plot dendrogram
    # plt.figure(figsize=(10, 7))
    # ac_plot_dendrogram(model, truncate_mode='level', p=3, color_threshold=distance_threshold)
    # plt.title("Agglomerative Clustering Dendrogram")
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()

    # Step 3b: Summarize the clusters
    cluster_summaries = ac_summarize_clusters(data, labels)

    # Step 4: Format the summaries fodefr readability
    formatted_summaries = ac_format_cluster_summary(cluster_summaries)

    # Print the formatted summaries
    for summary in formatted_summaries:
        print(summary)

# =======================
# C4.5 Method
# =======================
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from collections import defaultdict

def c4_5_preprocess_holes(holes):
    """Convert holes to a feature matrix and label vector."""
    features = []
    for hole in holes:
        row = []
        for element in hole:
            if isinstance(element, tuple):
                row.append(list(element))
            else:
                row.append([element])
        features.append(row)
    
    # Flatten nested lists for sklearn compatibility
    X = []
    for feature in features:
        X.append([item for sublist in feature for item in sublist])
    
    # Generate labels (placeholders, as we are not doing classification)
    y = np.zeros(len(holes))
    
    return np.array(X), y

def c4_5_train_decision_tree(X, y):
    """Train a decision tree on the preprocessed data."""
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X, y)
    return clf

def c4_5_cluster_holes(clf, X):
    """Cluster holes based on decision tree leaves."""
    leaf_indices = clf.apply(X)
    clusters = defaultdict(list)
    
    for i, leaf_index in enumerate(leaf_indices):
        print(leaf_index)
        clusters[leaf_index].append(X[i])
    
    return clusters

def c4_5_summarize_cluster(cluster):
    """Summarize a cluster of combinations."""
    summary = []
    num_columns = len(cluster[0])
    for i in range(num_columns):
        column_values = [item[i] for item in cluster]
        unique_values = set(column_values)
        if len(unique_values) == 1:
            summary.append(list(unique_values)[0])
        else:
            summary.append(list(unique_values))
    return summary

def c4_5_format_cluster_summary(clusters):
    """Format the cluster summaries into a readable format."""
    formatted_summaries = []
    for i, (leaf, cluster) in enumerate(clusters.items()):
        # print(i, leaf, cluster)
        summary = c4_5_summarize_cluster(cluster)
        formatted_summary = []
        for item in summary:
            if isinstance(item, list):
                formatted_summary.append(f"[{', '.join(map(str, item))}]")
            else:
                formatted_summary.append(str(item))
        formatted_summaries.append(f"Cluster {i}: {', '.join(formatted_summary)}")
    return formatted_summaries

def c4_5_summary(holes):
    # Preprocess the data
    X, y = c4_5_preprocess_holes(holes)

    # Train the decision tree
    clf = c4_5_train_decision_tree(X, y)

    # Cluster the holes
    clusters = c4_5_cluster_holes(clf, X)

    # Format and print the clusters
    formatted_summaries = c4_5_format_cluster_summary(clusters)
    for summary in formatted_summaries:
        print(summary)


# Example usage
if __name__ == "__main__":
    config_path = 'config.json'  # Path to JSON config file
    data_util = DataUtil(config_path)

    db_path = data_util.db_name
    table_name = "data"
    
    missing_combinations = find_missing_combinations(db_path, table_name, data_util)
    # missing_combinations.append((0,3,2))

    # Calculate total number of possible combinations
    total_combinations = data_util.generate_total_number_of_combinations()

    print(f"Missing combinations: {len(missing_combinations)} ({(len(missing_combinations)/total_combinations)*100:.2f}%)")
    # print(missing_combinations)

    # # Example usage
    # missing_combinations = [
    #     (1, 0, 0, 0),
    #     (2, 0, 0, 0),
    #     (1, 1, 1, 1),
    #     (1, 1, 1, 2),
    #     (1, 1, 1, 3),
    #     (1, 2, 2, 1),
    #     (1, 2, 2, 2)
    # ]

    # with Timer() as timer:
    #     # Hamming Distance Method
    #     aggregated_holes = iterative_aggregation(missing_combinations)
    #     # summary = summarize_aggregated_holes(aggregated_holes)
    #     print_sorted_holes(sort_holes_by_size(aggregated_holes))

    ac_summary(missing_combinations, True)
    # c4_5_summary(missing_combinations)
