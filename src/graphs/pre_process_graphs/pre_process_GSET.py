import networkx as nx
import os
import pickle
import json
def read_gset(file_path):
    """
    Reads a Gset-format file and returns a weighted NetworkX Graph.

    Parameters
    ----------
    file_path : str
        Path to the Gset file.

    Returns
    -------
    G : networkx.Graph
        A Graph with N nodes labeled 0..N, and E edges with 'weight' attributes.
    """
    try:
        with open(file_path, 'r') as f:
            # Read the first line to get N and E
            first_line = f.readline().strip()
            N = int(first_line.split()[0]) 
            
            # Initialize an empty undirected Graph
            G = nx.Graph()
            
            # Add all nodes (0-based indexing as the file types are like that)
            G.add_nodes_from(range(0,N))
            
            # Iterate over the remaining lines, each representing an edge
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines if any
                
                # Split the line and extract edge data
                i, j, w = line.split()
                i, j, w = int(i), int(j), float(w)
                
                # Add the edge to the graph with its weight
                G.add_edge(i, j, weight=w)
                
        return G
    except Exception as e:
        print("Error in reading the file: ", file_path)
        print(e)
        exit()
        return None

if __name__ == "__main__":

    Gset_folder = '/storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/data/real_world/stanford_GSET'


    json_file = Gset_folder+ '/final_sizes.json'
    # delete if exists
    if os.path.exists(json_file):
        os.remove(json_file)
        print("Deleted the final json file")


    # Loop over all files in the folder
    files = os.listdir(Gset_folder)
    files = [f for f in files if f.endswith('.gpickle')]
    
    # delete all the gpickle files
    if len(files) > 0:
        for file in files:
            os.remove(Gset_folder + '/' + file)
        print("Deleted all the gpickle files")
 

    files = os.listdir(Gset_folder)
    files = [f for f in files if not f.endswith('.gpickle')]
    files = sorted(files)
 

    final_json = {}

    # Create a graph for each file
    for file in files:
        file_path = os.path.join(Gset_folder, file)
        graph = read_gset(file_path)
        if graph is None:
            continue
        else:
            final_json[file] = { 
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges()
            }
            with open(Gset_folder+ '/{}.gpickle'.format(file), 'wb') as f:
                pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
        
    
    # Save the final json
    with open(json_file, 'w') as f:
        json.dump(final_json, f)