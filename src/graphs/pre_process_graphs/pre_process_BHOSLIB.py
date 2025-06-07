import networkx as nx
import os
import pickle
import json
def read_clq(file_path):
    """
    Reads a BHOSLIB .clq file and returns an unweighted NetworkX Graph.

    File format:
      - The first line is a header of the form:
            p edge <num_nodes> <num_edges>
      - Each subsequent line starting with 'e' represents an edge:
            e <node1> <node2>
        where nodes are 1-indexed.
    
    Parameters
    ----------
    file_path : str
        Path to the .clq file.
    
    Returns
    -------
    G : networkx.Graph
        A graph with nodes 1 through <num_nodes> and the specified edges.
    """
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Process the header line
            if line.startswith('p'):
                # Header line should be: p edge <num_nodes> <num_edges>
                parts = line.split()
                if len(parts) >= 4:
                    num_nodes = int(parts[2])
                    # num_edges = int(parts[3])  # You can use this if needed for verification.
                    # Add nodes to the graph (1-indexed)
                    G.add_nodes_from(range(num_nodes))
            # Process an edge line
            elif line.startswith('e'):
                parts = line.split()
                if len(parts) >= 3:
                    u = int(parts[1])-1
                    v = int(parts[2])-1
                    G.add_edge(u, v)
    return G

if __name__ == "__main__":

    data_folders = ['/storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/data/real_world/BHOSLIB-Max-Clique',
                    '/storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/data/real_world/DIMACS-Max-Clique']
    for folder in data_folders:
        json_file = folder+ '/final_sizes.json'
        # delete if exists
        if os.path.exists(json_file):
            os.remove(json_file)
            print("Deleted the final json file")


        # Loop over all files in the folder
        files = os.listdir(folder)
        files = [f for f in files if f.endswith('.gpickle')]
        # delete all the gpickle files
        if len(files) > 0:
            for file in files:
                os.remove(folder + '/' + file)
            print("Deleted all the gpickle files")

        files = os.listdir(folder)
        files = [f for f in files if f.endswith('.clq')]
        files = sorted(files)


        final_json = {}

        # Create a graph for each file
        for file in files:
            file_path = os.path.join(folder, file)
            graph = read_clq(file_path)
            if graph is None:
                continue
            else:
                final_json[file] = { 
                    'nodes': graph.number_of_nodes(),
                    'edges': graph.number_of_edges()
                }
                with open(folder+ '/{}.gpickle'.format(file), 'wb') as f:
                    pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

        # Save the final json
        with open(json_file, 'w') as f:
            json.dump(final_json, f)