# Problem 1

# Equivalent Resistance Using Graph Theory

## Problem Description

**Motivation:**
Calculating equivalent resistance is a fundamental problem in electrical circuits, essential for understanding and designing efficient systems. While traditional methods involve iteratively applying series and parallel resistor rules, these approaches can become cumbersome for complex circuits with many components. Graph theory offers a powerful alternative, providing a structured and algorithmic way to analyze circuits.

By representing a circuit as a graph—where nodes correspond to junctions and edges represent resistors with weights equal to their resistance values—we can systematically simplify even the most intricate networks. This method not only streamlines calculations but also opens the door to automated analysis, making it particularly useful in modern applications like circuit simulation software, optimization problems, and network design.

Studying equivalent resistance through graph theory is valuable not only for its practical applications but also for the deeper insights it provides into the interplay between electrical and mathematical concepts. This approach highlights the versatility of graph theory, demonstrating its relevance across physics, engineering, and computer science.

## Theory and Approach

When analyzing electrical circuits using graph theory:
- Nodes represent junctions
- Edges represent resistors (with weights equal to resistance values)

The key insight is that we can systematically simplify the circuit graph by identifying and reducing:
1. Series connections - resistors connected in a chain
2. Parallel connections - resistors connecting the same two nodes

## Implementation

```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class EquivalentResistanceCalculator:
    def __init__(self):
        self.step_count = 0
        self.debug = False
        
    def set_debug(self, debug=True):
        """Enable or disable debug mode."""
        self.debug = debug
    
    def calculate_equivalent_resistance(self, graph, source, target):
        """
        Calculate the equivalent resistance between source and target nodes in the given graph.
        
        Args:
            graph: A NetworkX graph where edges have 'resistance' attribute
            source: Source node
            target: Target node
            
        Returns:
            The equivalent resistance between source and target
        """
        # Create a working copy of the graph
        G = graph.copy()
        
        if self.debug:
            print(f"Initial graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
            self._visualize_graph(G, source, target, "Initial Circuit")
        
        # Continue simplifying until only source and target remain
        while len(G.nodes) > 2:
            self.step_count += 1
            
            if self.debug:
                print(f"\nStep {self.step_count}:")
            
            # Try parallel reduction first
            if self._reduce_parallel(G):
                if self.debug:
                    print("  Performed parallel reduction")
                    if len(G.nodes) <= 10:  # Only visualize for reasonably sized graphs
                        self._visualize_graph(G, source, target, f"After Parallel Reduction - Step {self.step_count}")
                continue
                
            # Try series reduction
            if self._reduce_series(G, source, target):
                if self.debug:
                    print("  Performed series reduction")
                    if len(G.nodes) <= 10:
                        self._visualize_graph(G, source, target, f"After Series Reduction - Step {self.step_count}")
                continue
            
            # If we can't simplify further with series or parallel, we need to use Y-Δ transformation
            if self._apply_y_delta_transformation(G, source, target):
                if self.debug:
                    print("  Performed Y-Δ transformation")
                    if len(G.nodes) <= 10:
                        self._visualize_graph(G, source, target, f"After Y-Δ Transformation - Step {self.step_count}")
                continue
                
            # If no reduction was made, use node elimination method for complex circuits
            if self._node_elimination(G, source, target):
                if self.debug:
                    print("  Performed node elimination")
                    if len(G.nodes) <= 10:
                        self._visualize_graph(G, source, target, f"After Node Elimination - Step {self.step_count}")
                continue
                
            # If we reach here, there's a problem with the circuit
            raise ValueError("Could not simplify the circuit further. Check circuit topology.")
        
        # At this point, there should be just one edge between source and target
        if len(G.edges) != 1:
            raise ValueError("Circuit could not be reduced properly.")
            
        # Get the final resistance
        edge_data = G.get_edge_data(source, target)
        if edge_data is None:
            raise ValueError(f"No connection between {source} and {target}")
            
        equivalent_resistance = edge_data['resistance']
        
        if self.debug:
            print(f"\nFinal equivalent resistance: {equivalent_resistance}")
            
        return equivalent_resistance
    
    def _reduce_parallel(self, G):
        """
        Find and reduce parallel resistors in the graph.
        Returns True if a reduction was performed, False otherwise.
        """
        # Find all pairs of nodes that have multiple edges between them
        for u in list(G.nodes()):
            neighbors = list(G.neighbors(u))
            
            for v in neighbors:
                # Check if there are multiple edges between u and v
                parallel_edges = list(G.edges(nbunch=[u], data=True))
                parallel_edges = [e for e in parallel_edges if e[1] == v]
                
                if len(parallel_edges) > 1:
                    # Calculate equivalent resistance for parallel resistors: 1/Req = 1/R1 + 1/R2 + ...
                    conductance_sum = sum(1 / e[2]['resistance'] for e in parallel_edges)
                    equivalent_resistance = 1 / conductance_sum
                    
                    # Remove all parallel edges
                    for edge in parallel_edges:
                        G.remove_edge(edge[0], edge[1])
                    
                    # Add new edge with equivalent resistance
                    G.add_edge(u, v, resistance=equivalent_resistance)
                    
                    if self.debug:
                        resistances = [f"{e[2]['resistance']:.4f}" for e in parallel_edges]
                        print(f"  Reduced parallel resistors between {u}-{v}: {', '.join(resistances)} → {equivalent_resistance:.4f}")
                    
                    return True
                    
        return False
    
    def _reduce_series(self, G, source, target):
        """
        Find and reduce series resistors in the graph.
        Returns True if a reduction was performed, False otherwise.
        """
        # Look for nodes with exactly 2 connections that are not source or target
        for node in list(G.nodes()):
            if node == source or node == target:
                continue
                
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 2:
                n1, n2 = neighbors
                
                # Get the resistances
                r1 = G.get_edge_data(node, n1)['resistance']
                r2 = G.get_edge_data(node, n2)['resistance']
                
                # Calculate series equivalent: Req = R1 + R2
                equivalent_resistance = r1 + r2
                
                # Remove the intermediate node and its edges
                G.remove_node(node)
                
                # Add new direct connection between the neighbors
                G.add_edge(n1, n2, resistance=equivalent_resistance)
                
                if self.debug:
                    print(f"  Reduced series resistors at node {node}: {r1:.4f} + {r2:.4f} → {equivalent_resistance:.4f}")
                
                return True
                
        return False
    
    def _apply_y_delta_transformation(self, G, source, target):
        """
        Perform a Y-Δ transformation when possible.
        This transforms a Y configuration (star) into a Δ configuration (delta/triangle).
        Returns True if a transformation was made, False otherwise.
        """
        # Look for Y configurations (a node connected to exactly 3 other nodes)
        for node in list(G.nodes()):
            if node == source or node == target:
                continue
                
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 3:
                # We have a Y configuration centered at 'node'
                a, b, c = neighbors
                
                # Get the resistance values of the three branches
                r1 = G.get_edge_data(node, a)['resistance']  # Between node and a
                r2 = G.get_edge_data(node, b)['resistance']  # Between node and b
                r3 = G.get_edge_data(node, c)['resistance']  # Between node and c
                
                # Calculate the delta (triangle) equivalent resistances
                sum_product = r1*r2 + r2*r3 + r3*r1
                r_ab = sum_product / r3  # Between a and b
                r_bc = sum_product / r1  # Between b and c
                r_ca = sum_product / r2  # Between c and a
                
                # Remove the Y center node and its edges
                G.remove_node(node)
                
                # Add the delta edges (if they don't already exist)
                if not G.has_edge(a, b):
                    G.add_edge(a, b, resistance=r_ab)
                else:
                    # If edge already exists, combine in parallel
                    existing_r = G.get_edge_data(a, b)['resistance']
                    G.add_edge(a, b, resistance=(existing_r * r_ab) / (existing_r + r_ab))
                    
                if not G.has_edge(b, c):
                    G.add_edge(b, c, resistance=r_bc)
                else:
                    existing_r = G.get_edge_data(b, c)['resistance']
                    G.add_edge(b, c, resistance=(existing_r * r_bc) / (existing_r + r_bc))
                    
                if not G.has_edge(c, a):
                    G.add_edge(c, a, resistance=r_ca)
                else:
                    existing_r = G.get_edge_data(c, a)['resistance']
                    G.add_edge(c, a, resistance=(existing_r * r_ca) / (existing_r + r_ca))
                
                if self.debug:
                    print(f"  Y-Δ transformation at node {node}: Y({r1:.4f}, {r2:.4f}, {r3:.4f}) → Δ({r_ab:.4f}, {r_bc:.4f}, {r_ca:.4f})")
                
                return True
                
        return False
    
    def _node_elimination(self, G, source, target):
        """
        Use node elimination method (similar to Gaussian elimination) for complex circuits.
        This method removes one node at a time and recalculates the equivalent circuit.
        Returns True if a node was eliminated, False otherwise.
        """
        for node in list(G.nodes()):
            if node == source or node == target:
                continue
                
            # Find all neighbors of this node
            neighbors = list(G.neighbors(node))
            if len(neighbors) < 2:
                continue  # Not enough connections to eliminate
                
            # Build the conductance matrix for this node and its neighbors
            conductances = {}
            for i in neighbors:
                r_i = G.get_edge_data(node, i)['resistance']
                conductances[(node, i)] = 1.0 / r_i
                
            # Calculate new connections between all pairs of neighbors
            for i in neighbors:
                for j in neighbors:
                    if i >= j:  # Avoid duplicate work
                        continue
                        
                    g_i = conductances[(node, i)]
                    g_j = conductances[(node, j)]
                    
                    # Calculate new resistance between i and j
                    if g_i + g_j > 0:  # Avoid division by zero
                        new_resistance = 1.0 / (g_i * g_j / sum(conductances.values()))
                        
                        # Add or update the connection
                        if G.has_edge(i, j):
                            old_r = G.get_edge_data(i, j)['resistance']
                            equivalent_r = (old_r * new_resistance) / (old_r + new_resistance)  # Parallel combination
                            G.add_edge(i, j, resistance=equivalent_r)
                        else:
                            G.add_edge(i, j, resistance=new_resistance)
            
            # Remove the node we're eliminating
            G.remove_node(node)
            
            if self.debug:
                print(f"  Node elimination: Removed node {node}")
                
            return True
            
        return False
    
    def _visualize_graph(self, G, source, target, title="Circuit Graph"):
        """Visualize the current state of the graph."""
        plt.figure(figsize=(10, 6))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500)
        
        # Highlight source and target
        nx.draw_networkx_nodes(G, pos, nodelist=[source, target], 
                              node_color='lightgreen', node_size=700)
        
        # Draw edges with resistance values as labels
        edge_labels = {(u, v): f"{d['resistance']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edges(G, pos, width=2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def create_example_circuit_1():
    """Simple series-parallel circuit."""
    G = nx.Graph()
    # Add resistors (edges with resistance attribute)
    G.add_edge('A', 'B', resistance=10.0)  # R1
    G.add_edge('B', 'C', resistance=20.0)  # R2
    G.add_edge('A', 'D', resistance=30.0)  # R3
    G.add_edge('D', 'C', resistance=40.0)  # R4
    
    return G, 'A', 'C'

def create_example_circuit_2():
    """Wheatstone bridge circuit."""
    G = nx.Graph()
    G.add_edge('A', 'B', resistance=10.0)
    G.add_edge('B', 'C', resistance=20.0)
    G.add_edge('A', 'D', resistance=30.0)
    G.add_edge('D', 'C', resistance=40.0)
    G.add_edge('B', 'D', resistance=50.0)  # Bridge resistor
    
    return G, 'A', 'C'

def create_example_circuit_3():
    """Complex circuit with multiple paths and nodes."""
    G = nx.Graph()
    G.add_edge('A', 'B', resistance=5.0)
    G.add_edge('B', 'C', resistance=10.0)
    G.add_edge('C', 'D', resistance=15.0)
    G.add_edge('D', 'E', resistance=20.0)
    G.add_edge('A', 'F', resistance=25.0)
    G.add_edge('F', 'G', resistance=30.0)
    G.add_edge('G', 'E', resistance=35.0)
    G.add_edge('B', 'G', resistance=40.0)
    G.add_edge('C', 'F', resistance=45.0)
    
    return G, 'A', 'E'

def run_examples():
    calculator = EquivalentResistanceCalculator()
    calculator.set_debug(True)
    
    # Example 1: Simple series-parallel circuit
    print("\n=== EXAMPLE 1: SIMPLE SERIES-PARALLEL CIRCUIT ===")
    circuit1, source1, target1 = create_example_circuit_1()
    calculator.step_count = 0
    equiv_resistance1 = calculator.calculate_equivalent_resistance(circuit1, source1, target1)
    print(f"Equivalent resistance for example 1: {equiv_resistance1:.4f} ohms")
    
    # Example 2: Wheatstone bridge circuit
    print("\n=== EXAMPLE 2: WHEATSTONE BRIDGE CIRCUIT ===")
    circuit2, source2, target2 = create_example_circuit_2()
    calculator.step_count = 0
    equiv_resistance2 = calculator.calculate_equivalent_resistance(circuit2, source2, target2)
    print(f"Equivalent resistance for example 2: {equiv_resistance2:.4f} ohms")
    
    # Example 3: Complex circuit
    print("\n=== EXAMPLE 3: COMPLEX CIRCUIT ===")
    circuit3, source3, target3 = create_example_circuit_3()
    calculator.step_count = 0
    equiv_resistance3 = calculator.calculate_equivalent_resistance(circuit3, source3, target3)
    print(f"Equivalent resistance for example 3: {equiv_resistance3:.4f} ohms")

# Run the examples if this script is executed
if __name__ == "__main__":
    run_examples()
```
![alt text](<Screenshot 2025-04-06 at 17.16.50.png>)
![alt text](<Screenshot 2025-04-06 at 17.17.05.png>)
![alt text](<Screenshot 2025-04-06 at 17.17.16.png>)
![alt text](<Screenshot 2025-04-06 at 17.17.32.png>)
![alt text](<Screenshot 2025-04-06 at 17.17.43.png>)
![alt text](<Screenshot 2025-04-06 at 17.17.54.png>)
![alt text](<Screenshot 2025-04-06 at 17.18.18.png>)
![alt text](<Screenshot 2025-04-06 at 17.18.28.png>)
![alt text](<Screenshot 2025-04-06 at 17.18.39.png>)
![alt text](<Screenshot 2025-04-06 at 17.18.49.png>)
![alt text](<Screenshot 2025-04-06 at 17.18.58.png>)
![alt text](<Screenshot 2025-04-06 at 17.23.36.png>)


## Algorithm Explanation

The implementation uses a systematic approach to simplify electrical circuits:

1. **Graph Representation**:
   - Each node represents a junction in the circuit
   - Each edge represents a resistor with weight equal to its resistance value

2. **Simplification Process**:
   - The algorithm iteratively simplifies the circuit until only source and target nodes remain
   - Two basic operations are performed:
     - **Parallel Reduction**: Multiple edges between the same pair of nodes are combined using the parallel resistor formula (1/Req = 1/R1 + 1/R2 + ...)
     - **Series Reduction**: Intermediate nodes with exactly two connections are eliminated, combining the resistors using the series formula (Req = R1 + R2)

3. **Termination**:
   - When only source and target nodes remain with a single edge between them, that edge's resistance is the equivalent resistance of the entire circuit

## Analysis of Example Cases

### Example 1: Simple Series-Parallel Circuit
- Four resistors arranged in a diamond pattern between nodes A and C
- The algorithm first reduces the parallel paths, then combines the series resistors
- This demonstrates basic series-parallel simplification

### Example 2: Wheatstone Bridge Circuit
- Similar to Example 1 but with an additional resistor connecting the middle nodes
- This creates a more complex topology that requires multiple reduction steps
- The algorithm handles this by systematically applying parallel and series reductions

### Example 3: Complex Circuit
- A network with 7 nodes and 9 resistors with multiple paths between source and target
- Demonstrates the algorithm's ability to handle arbitrary complex topologies
- The nested combinations of series and parallel connections are systematically reduced

## Algorithm Efficiency and Potential Improvements

### Time Complexity
- The algorithm has worst-case time complexity of O(n²), where n is the number of nodes
- Each reduction step requires scanning all nodes and edges (O(n+e))
- Maximum number of reduction steps is O(n) since each step reduces node count by at least 1

### Space Complexity
- Space complexity is O(n+e) for storing the graph

### Potential Improvements:
1. **Optimization**: For very large circuits, the graph scanning could be optimized by keeping track of candidate nodes for reduction
2. **Matrix Methods**: For highly connected graphs, using nodal analysis with matrix methods (Kirchhoff's laws) could be more efficient
3. **Parallelization**: For extremely large networks, certain graph operations could be parallelized
4. **Special Case Handling**: Adding specialized handlers for common circuit topologies (like ladder networks or star networks)
5. **Numerical Stability**: For circuits with very large or very small resistance values, numerical precision improvements could be added

## Conclusion

Graph theory provides an elegant and systematic approach to calculating equivalent resistance in electrical circuits. The implemented algorithm successfully handles arbitrary circuit configurations by iteratively applying series and parallel reduction rules. It works efficiently for most practical circuits and can be extended to handle special cases or extremely large networks if needed.

The visualization capabilities included in the implementation make it useful not just for calculation but also as an educational tool to understand how complex circuits can be systematically simplified.