import networkx as nx
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple

def load_front_dataset(filepath: str) -> List[Dict]:
    """
    Load the FRONT dataset from JSON file.
    
    Args:
        filepath: Path to the relationships_anyscene.json file
    
    Returns:
        List of scene dictionaries
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data.get('scans', [])
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return []

def extract_scene_graph(scene: Dict) -> Tuple[Dict, List]:
    """
    Extract objects and relationships from a scene.
    
    Args:
        scene: Dictionary containing scene data
    
    Returns:
        Tuple of (objects dictionary, relationships list)
    """
    return scene.get('objects', {}), scene.get('relationships', [])

def create_scene_graph(objects: Dict, relationships: List) -> nx.DiGraph:
    """Create directed graph from objects and relationships."""
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes (objects)
    for obj_id, obj_name in objects.items():
        if obj_name != "floor":  # Ignore floor
            G.add_node(obj_id, name=obj_name)
    
    # Define allowed relationship types
    allowed_relations = {
        "left", "right", "behind", "front",
        "bigger than", "lower than", "higher than"
    }
    
    # Add edges (relationships)
    for rel in relationships:
        obj1, obj2, _, rel_type = rel
        # Convert to strings since the objects dict uses strings as keys
        obj1, obj2 = str(obj1), str(obj2)
        
        # Skip if either object is floor (8)
        if obj1 == "8" or obj2 == "8":
            continue
            
        # Only add edges for allowed relationship types
        if rel_type in allowed_relations:
            G.add_edge(obj1, obj2, relationship=rel_type)
    
    return G

def visualize_graph(G: nx.DiGraph, scene_id: str = ""):
    """
    Visualize the scene graph.
    
    Args:
        G: NetworkX directed graph
        scene_id: Optional scene identifier for the title
    """
    plt.figure(figsize=(12, 8))
    
    # Use spring layout for node positioning
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.7)
    
    # Draw node labels (object names)
    labels = nx.get_node_attributes(G, 'name')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Draw edges with different colors based on relationship type
    edge_colors = {
        "left": "red",
        "right": "blue",
        "behind": "green",
        "front": "purple",
        "bigger than": "orange",
        "lower than": "brown",
        "higher than": "pink"
    }
    
    # Draw edges for each relationship type
    for rel_type, color in edge_colors.items():
        edges = [(u, v) for (u, v, d) in G.edges(data=True) 
                if d['relationship'] == rel_type]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, 
                             arrows=True, arrowsize=20)
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], color=color, label=rel_type)
                      for rel_type, color in edge_colors.items()]
    plt.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1, 0.5))
    
    title = f"Scene Graph Visualization{' - ' + scene_id if scene_id else ''}"
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    output_path = f"../scenegraphvisuals/{scene_id}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close() 

def main():
    # Load dataset
    filepath = "/home/ubuntu/datasets/FRONT/relationships_anyscene.json"
    scenes = load_front_dataset(filepath)
    
    if not scenes:
        print("No scenes found in dataset")
        return
    
    # Process each scene
    for scene in scenes:
        # Extract scene ID for the title
        scene_id = scene.get('scan', 'Unknown Scene')
        
        # Extract objects and relationships
        objects, relationships = extract_scene_graph(scene)
        
        # Create and visualize graph
        G = create_scene_graph(objects, relationships)
        
        if G.number_of_nodes() > 0:  # Only visualize if there are nodes
            visualize_graph(G, scene_id)
            
            # Optional: Ask user if they want to continue to next scene
            response = input("Press Enter to continue to next scene, or 'q' to quit: ")
            if response.lower() == 'q':
                break
        else:
            print(f"No valid nodes found in scene {scene_id}")

if __name__ == "__main__":
    main()
