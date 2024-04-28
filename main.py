import networkx as nx
import igraph as ig
import re
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics.cluster import contingency_matrix, normalized_mutual_info_score, adjusted_mutual_info_score

NETWORKS_FOLDER = Path("networks")
CLUSTERING_ALGORITHM = "infomap"
results = []

def apply_algorithms(G, algorithm):
    g = ig.Graph.TupleList(G.edges())

    match algorithm:
        case "infomap": 
            return g.community_infomap()
        
        case "louvain":
            return g.community_multilevel()
    
        case "leiden":
            return g.community_leiden()

        case _:
            print("No matching algorithm was provided")

def extract_prr(file_name):
    prr = re.search(r"prr_([0-9.]+)", file_name)
    
    return float(prr.group(1))

def evaluate_networks(cluster, num_nodes=300, group_size=60):
    true_labels = [i // group_size for i in range(num_nodes)]
    matrix = contingency_matrix(true_labels, cluster.membership)
    nmi = normalized_mutual_info_score(true_labels, cluster.membership)
    ami = adjusted_mutual_info_score(true_labels, cluster.membership)
    jaccard = _jaccard_index_clusters(true_labels, cluster.membership)

    return matrix, nmi, ami, jaccard

def _jaccard_index_clusters(true_labels, predicted_labels):
    true_clusters = {label: set() for label in set(true_labels)}
    predicted_clusters = {label: set() for label in set(predicted_labels)}

    for idx, label in enumerate(true_labels):
        true_clusters[label].add(idx)
    for idx, label in enumerate(predicted_labels):
        predicted_clusters[label].add(idx)

    intersections, unions = 0,0
    for true_set in true_clusters.values():
        for predicted_set in predicted_clusters.values():
            intersection = len(true_set & predicted_set)
            union = len(true_set | predicted_set)
            intersections += intersection
            unions += union

    return intersections / unions if unions != 0 else 0

def plot_results(results, algorithm):
    sorted_results = sorted(results, key=lambda x: x["prr"])
    prr = [result["prr"] for result in sorted_results]
    _plot_modularities_and_n_clusters(sorted_results, prr, algorithm)
    _plot_metrics(sorted_results, prr, algorithm)

def _plot_modularities_and_n_clusters(sorted_results, prr, algorithm):
    modularities = [result["modularity"] for result in sorted_results]
    n_communities = [result["number_of_communities"] for result in sorted_results]

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(prr, modularities, linestyle="-", marker="o")
    plt.title("Modularity by PRR")
    plt.xlabel("PRR")
    plt.ylabel("Modularity")

    plt.subplot(1,2,2)
    plt.plot(prr, n_communities, linestyle="-", marker="o")
    plt.title("Number of Communities by PRR")
    plt.xlabel("PRR")
    plt.ylabel("Number of Communities")

    plt.suptitle(f"Results for the {algorithm} algorithm")
    plt.tight_layout()
    plt.show()

def _plot_metrics(sorted_results, prr, algorithm):
    jaccard = [result["jaccard"] for result in sorted_results]
    nmi = [result["nmi"] for result in sorted_results]
    ami = [result["ami"] for result in sorted_results]

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(prr, jaccard, linestyle="-", marker="o")
    plt.title("Jaccard index by PRR")
    plt.xlabel("PRR")
    plt.ylabel("Jaccard Index")

    plt.subplot(1,3,2)
    plt.plot(prr, nmi, linestyle="-", marker="o")
    plt.title("Normalized Mutual Information by PRR")
    plt.xlabel("PRR")
    plt.ylabel("NMI")

    plt.subplot(1,3,3)
    plt.plot(prr, ami, linestyle="-", marker="o")
    plt.title("Adjusted Mutual Information by PRR")
    plt.xlabel("PRR")
    plt.ylabel("AMI")

    plt.suptitle(f"Results for the {algorithm} algorithm")
    plt.tight_layout()
    plt.show()

for network in NETWORKS_FOLDER.glob("*.net"):
    try:
        G = nx.read_pajek(network)
        cluster = apply_algorithms(G, CLUSTERING_ALGORITHM)
        prr = extract_prr(str(network))
        matrix, nmi, ami, jaccard = evaluate_networks(cluster)

        results.append({
            "filename": network,
            "clusters": cluster,
            "clustering_algorithm": CLUSTERING_ALGORITHM,
            "prr": prr,
            "modularity": cluster.modularity,
            "number_of_communities": len(cluster),
            "contingency_matrix": matrix,
            "nmi":nmi,
            "ami":ami,
            "jaccard":jaccard
        })

    except Exception as e:
        print("Error loading the file:", e)

plot_results(results, CLUSTERING_ALGORITHM)
