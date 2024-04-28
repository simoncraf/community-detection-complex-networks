import networkx as nx

# Attempt to load the network
file_path = "networks\synthetic_network_N_300_blocks_5_prr_0.06_prs_0.02.net"
try:
    G = nx.read_pajek(file_path)
    print("Loaded successfully!")
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
except Exception as e:
    print("Error loading the file:", e)
