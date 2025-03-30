# Problem 1

# Water Wave Interference Patterns Analysis

## Introduction

This document presents a comprehensive analysis of interference patterns formed by water waves emanating from point sources positioned at the vertices of regular polygons. Water wave interference is a fascinating example of wave superposition that demonstrates fundamental physical principles in a visual and intuitive way.

## Theoretical Background

### Single Disturbance Equation

A circular wave emanating from a point source located at position (x₀, y₀) can be described by:

$$\eta(x, y, t) = A \cos(kr - \omega t + \phi)$$

Where:
- $\eta(x, y, t)$ is the displacement of the water surface at point $(x, y)$ and time $t$
- $A$ is the amplitude of the wave
- $k$ is the wave number, related to the wavelength $\lambda$ by $k = 2\pi/\lambda$
- $\omega$ is the angular frequency, related to the frequency $f$ by $\omega = 2\pi f$
- $r$ is the distance from the source to the point $(x, y)$: $r = \sqrt{(x-x_0)^2 + (y-y_0)^2}$
- $\phi$ is the initial phase

### Principle of Superposition

When multiple waves overlap, the resulting displacement at any point is the algebraic sum of the individual displacements:

$$\eta_{total}(x, y, t) = \sum_{i=1}^{n} \eta_i(x, y, t)$$

Where $n$ is the number of sources (vertices of the polygon).

## Python Implementation

Below is the complete Python implementation for simulating and analyzing water wave interference patterns:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Define the single disturbance equation for a circular wave
def circular_wave(x, y, source_x, source_y, A, k, omega, t, phi=0):
    """
    Calculate the displacement of a circular wave at point (x, y) at time t.
    
    Parameters:
    x, y: Coordinates of the point
    source_x, source_y: Coordinates of the wave source
    A: Amplitude of the wave
    k: Wave number (k = 2π/λ)
    omega: Angular frequency (ω = 2πf)
    t: Time
    phi: Initial phase
    
    Returns:
    Displacement of the water surface
    """
    r = np.sqrt((x - source_x)**2 + (y - source_y)**2)
    return A * np.cos(k*r - omega*t + phi)

# Function to generate coordinates of vertices for a regular polygon
def regular_polygon_vertices(n, radius, center=(0, 0)):
    """
    Generate vertices of a regular polygon.
    
    Parameters:
    n: Number of sides (vertices)
    radius: Distance from center to vertices
    center: Center coordinates of the polygon
    
    Returns:
    List of (x, y) coordinates for each vertex
    """
    vertices = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        vertices.append((x, y))
    return vertices

# Function to calculate the superposition of waves from all sources
def calculate_superposition(x_grid, y_grid, sources, A, k, omega, t):
    """
    Calculate the superposition of waves from multiple sources.
    
    Parameters:
    x_grid, y_grid: Meshgrid of x, y coordinates
    sources: List of (x, y) coordinates of wave sources
    A, k, omega: Wave parameters
    t: Time
    
    Returns:
    Total displacement at each point in the grid
    """
    total = np.zeros_like(x_grid)
    for source_x, source_y in sources:
        total += circular_wave(x_grid, y_grid, source_x, source_y, A, k, omega, t)
    return total

# Main simulation function
def simulate_interference_patterns(polygon_sides=3, simulation_size=10, resolution=500, 
                                   polygon_radius=2, wave_amplitude=1, wavelength=1, 
                                   frequency=1, num_frames=60, animation_duration=5):
    """
    Simulate and visualize interference patterns from sources at polygon vertices.
    
    Parameters:
    polygon_sides: Number of sides of the regular polygon
    simulation_size: Size of the simulation area (e.g., 10x10 units)
    resolution: Grid resolution (higher = more detailed)
    polygon_radius: Distance from center to vertices
    wave_amplitude: Amplitude of the waves (A)
    wavelength: Wavelength of the waves (λ)
    frequency: Frequency of the waves (f)
    num_frames: Number of frames for animation
    animation_duration: Duration of animation in seconds
    
    Returns:
    Figure, animation, and final frame data
    """
    # Compute wave parameters
    k = 2 * np.pi / wavelength  # Wave number
    omega = 2 * np.pi * frequency  # Angular frequency
    
    # Create a grid for the water surface
    x = np.linspace(-simulation_size/2, simulation_size/2, resolution)
    y = np.linspace(-simulation_size/2, simulation_size/2, resolution)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Generate polygon vertices as wave sources
    sources = regular_polygon_vertices(polygon_sides, polygon_radius)
    
    # Set up the figure for visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create 3D surface plot
    ax1 = axes[0]
    ax2 = plt.subplot(122, projection='3d')
    
    # Function to update the plot for each frame of the animation
    def update(frame):
        t = frame / num_frames * animation_duration
        
        # Calculate wave superposition at this time
        z = calculate_superposition(x_grid, y_grid, sources, wave_amplitude, k, omega, t)
        
        # Update the 2D heatmap with interference pattern
        ax1.clear()
        contour = ax1.imshow(z, extent=[-simulation_size/2, simulation_size/2, -simulation_size/2, simulation_size/2], 
                      cmap='RdBu', vmin=-wave_amplitude*polygon_sides, vmax=wave_amplitude*polygon_sides)
        ax1.set_title(f'Interference Pattern (t={t:.2f}s)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # Plot source positions
        for src_x, src_y in sources:
            ax1.plot(src_x, src_y, 'o', color='black', markersize=8)
        
        # Update the 3D surface plot
        ax2.clear()
        surf = ax2.plot_surface(x_grid, y_grid, z, cmap=cm.coolwarm, linewidth=0, 
                               antialiased=True, vmin=-wave_amplitude*polygon_sides, 
                               vmax=wave_amplitude*polygon_sides)
        ax2.set_title(f'3D Surface (t={t:.2f}s)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('Displacement')
        ax2.set_zlim(-wave_amplitude*polygon_sides, wave_amplitude*polygon_sides)
        
        return contour, surf
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=animation_duration*1000/num_frames, blit=False)
    
    # Calculate the final frame for static analysis
    final_t = animation_duration
    final_z = calculate_superposition(x_grid, y_grid, sources, wave_amplitude, k, omega, final_t)
    
    plt.tight_layout()
    return fig, ani, final_z, sources, x_grid, y_grid

# Function to analyze a single static frame of the interference pattern
def analyze_interference(x_grid, y_grid, z, sources, wave_amplitude, polygon_sides):
    """
    Analyze and visualize a single frame of the interference pattern.
    
    Parameters:
    x_grid, y_grid: Meshgrid of x, y coordinates
    z: Wave displacement values
    sources: List of source coordinates
    wave_amplitude: Amplitude of individual waves
    polygon_sides: Number of sides of the polygon
    
    Returns:
    Figure with analysis plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 2D heatmap of interference pattern
    contour = axes[0].imshow(z, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], 
                            cmap='RdBu', vmin=-wave_amplitude*polygon_sides, vmax=wave_amplitude*polygon_sides)
    axes[0].set_title('Interference Pattern')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    for src_x, src_y in sources:
        axes[0].plot(src_x, src_y, 'o', color='black', markersize=8)
    fig.colorbar(contour, ax=axes[0], label='Displacement')
    
    # Identify regions of constructive and destructive interference
    threshold = 0.8 * wave_amplitude * polygon_sides
    constructive = np.ma.masked_where(z < threshold, z)
    destructive = np.ma.masked_where(z > -threshold, z)
    
    # Plot constructive interference regions
    axes[1].imshow(constructive, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], 
                  cmap='Reds', vmin=-wave_amplitude*polygon_sides, vmax=wave_amplitude*polygon_sides)
    axes[1].set_title('Constructive Interference Regions')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    for src_x, src_y in sources:
        axes[1].plot(src_x, src_y, 'o', color='black', markersize=8)
    
    # Plot destructive interference regions
    axes[2].imshow(destructive, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], 
                  cmap='Blues_r', vmin=-wave_amplitude*polygon_sides, vmax=wave_amplitude*polygon_sides)
    axes[2].set_title('Destructive Interference Regions')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    for src_x, src_y in sources:
        axes[2].plot(src_x, src_y, 'o', color='black', markersize=8)
    
    plt.tight_layout()
    return fig

# Analyze different regular polygons
def compare_polygons(max_sides=5, simulation_size=10, resolution=300, polygon_radius=2, 
                    wave_amplitude=1, wavelength=1, frequency=1, time=0):
    """
    Compare interference patterns for different regular polygons.
    
    Parameters:
    max_sides: Maximum number of sides to analyze
    Other parameters: Same as in simulate_interference_patterns
    
    Returns:
    Figure with comparison plots
    """
    fig, axes = plt.subplots(2, max_sides, figsize=(4*max_sides, 8))
    
    # Wave parameters
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi * frequency
    
    # Create grid
    x = np.linspace(-simulation_size/2, simulation_size/2, resolution)
    y = np.linspace(-simulation_size/2, simulation_size/2, resolution)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Analyze each polygon
    for n in range(1, max_sides + 1):
        # Get sources
        sources = regular_polygon_vertices(n, polygon_radius)
        
        # Calculate superposition
        z = calculate_superposition(x_grid, y_grid, sources, wave_amplitude, k, omega, time)
        
        # Plot 2D interference pattern
        im = axes[0, n-1].imshow(z, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], 
                               cmap='RdBu', vmin=-wave_amplitude*n, vmax=wave_amplitude*n)
        axes[0, n-1].set_title(f'{n} Sources\n({"Point" if n==1 else "Line" if n==2 else "Triangle" if n==3 else "Square" if n==4 else "Pentagon"})')
        axes[0, n-1].set_xlabel('x')
        axes[0, n-1].set_ylabel('y')
        
        # Plot sources
        for src_x, src_y in sources:
            axes[0, n-1].plot(src_x, src_y, 'o', color='black', markersize=6)
        
        # Plot cross-section along y=0
        middle_row = resolution // 2
        axes[1, n-1].plot(x, z[middle_row, :])
        axes[1, n-1].set_title(f'Cross-section at y=0')
        axes[1, n-1].set_xlabel('x')
        axes[1, n-1].set_ylabel('Displacement')
        axes[1, n-1].grid(True)
        axes[1, n-1].set_ylim(-wave_amplitude*n, wave_amplitude*n)
    
    plt.tight_layout()
    return fig

# Example usage and demonstration
if __name__ == "__main__":
    # Parameters
    polygon_sides = 3  # Number of sources (triangle)
    simulation_size = 10  # Size of the simulation area
    resolution = 300  # Grid resolution
    polygon_radius = 2  # Distance from center to vertices
    wave_amplitude = 1
    wavelength = 1
    frequency = 1
    
    print("Simulating interference patterns for an equilateral triangle (3 sources)...")
    
    # Run the simulation
    fig, ani, final_z, sources, x_grid, y_grid = simulate_interference_patterns(
        polygon_sides=polygon_sides,
        simulation_size=simulation_size,
        resolution=resolution,
        polygon_radius=polygon_radius,
        wave_amplitude=wave_amplitude,
        wavelength=wavelength,
        frequency=frequency,
        num_frames=50,
        animation_duration=2
    )
    
    # Analyze a static frame
    analysis_fig = analyze_interference(
        x_grid, y_grid, final_z, sources, 
        wave_amplitude, polygon_sides
    )
    
    # Compare different polygons
    comparison_fig = compare_polygons(
        max_sides=5,
        simulation_size=simulation_size,
        resolution=resolution,
        polygon_radius=polygon_radius,
        wave_amplitude=wave_amplitude,
        wavelength=wavelength,
        frequency=frequency
    )
    
    # Save the animation as GIF (option for later use)
    # ani.save('wave_interference.gif', writer='pillow', fps=15)
    
    # Display figures
    plt.show()
    
    print("Simulation complete!")
```
![alt text](<download (3).png>)
## Detailed Analysis of Interference Patterns

### Methodology

For this analysis, we chose to focus on regular polygons with 1 to 5 sides:
1. Single point source (for reference)
2. Two sources (line segment)
3. Three sources (equilateral triangle)
4. Four sources (square)
5. Five sources (regular pentagon)

For each configuration, we:
- Positioned the sources at equal distances from the origin
- Assumed all sources emit waves with identical amplitude, wavelength, and frequency
- Applied the superposition principle to calculate the displacement at each point
- Identified regions of constructive and destructive interference
- Visualized the resulting patterns in 2D and 3D

### Simulation Parameters

In our simulation, we used the following parameters:
- Wave amplitude (A): 1 unit
- Wavelength (λ): 1 unit
- Frequency (f): 1 Hz
- Distance from center to polygon vertices: 2 units
- Simulation area: 10×10 square units

### Results by Polygon Type

#### Single Source (Point)

A single source produces concentric circular waves radiating outward. With just one source, there's no interference pattern—just the familiar ripple pattern that decreases in amplitude with distance from the source (due to the spreading of the wave energy).

#### Two Sources (Line)

With two sources, we observe:
- A series of hyperbolic nodal lines (where destructive interference occurs)
- Alternating bands of constructive and destructive interference perpendicular to the line connecting the sources
- The spacing between adjacent maxima is λ/2 along directions perpendicular to the source axis
- The pattern exhibits mirror symmetry along both the line connecting the sources and the perpendicular bisector

This pattern is analogous to Young's double-slit experiment in optics. Points where waves arrive with a path difference of nλ (where n is an integer) experience constructive interference, while points with a path difference of (n+½)λ experience destructive interference.

#### Three Sources (Equilateral Triangle)

With three sources arranged in an equilateral triangle, we observe:
- A complex hexagonal-like pattern with six-fold symmetry 
- Distinctive star-shaped regions of constructive interference
- Multiple nodal lines (regions of destructive interference) creating intricate patterns
- High-amplitude regions at the center where waves from all three sources can constructively interfere
- The pattern repeats radially with decreasing intensity as distance from the center increases

The triangular arrangement creates a beautiful pattern that reflects the geometric symmetry of the source configuration. The six-fold symmetry (rather than three-fold) occurs because each pair of sources creates its own interference pattern, and these patterns overlap.

#### Four Sources (Square)

With four sources arranged in a square, we observe:
- A pattern with four-fold rotational symmetry
- A grid-like interference pattern with consistent nodal spacing
- Strong constructive interference at the center and along certain radial directions
- More complex interaction regions farther from the sources
- Clear periodic structure in both x and y directions

The square arrangement produces more ordered patterns than the triangle, with perpendicular nodal lines that form a lattice-like structure. This greater regularity results from the higher symmetry of the square compared to the triangle.

#### Five Sources (Pentagon)

With five sources arranged in a regular pentagon, we observe:
- A star-like pattern with five-fold symmetry
- More densely packed nodal lines
- Complex regions of constructive interference that form pentagonal patterns
- Highly symmetric behavior that mirrors the geometry of the source arrangement
- A blend of order and complexity that creates visually striking patterns

The five-source arrangement demonstrates how increasing the number of coherent sources creates more intricate and detailed interference patterns.

## Key Observations and Physical Insights

### 1. Symmetry Relationship

The symmetry of the interference pattern directly reflects the symmetry of the source arrangement. An n-sided regular polygon produces patterns with n-fold rotational symmetry. This is a manifestation of the principle that the symmetry of a physical system is preserved in its solutions.

### 2. Constructive and Destructive Interference

- **Constructive Interference**: Occurs when waves arrive in phase, resulting in amplification. The maximum possible amplitude is n·A, where n is the number of sources and A is the amplitude of each wave.
- **Destructive Interference**: Occurs when waves arrive out of phase, resulting in cancellation. Complete destructive interference requires waves to arrive with exactly opposite phases.

### 3. Distance Effects

The interference pattern changes with distance from the source array:
- **Near Field**: Close to the sources, the pattern is dominated by the proximity to individual sources
- **Intermediate Field**: Complex interference patterns are most evident
- **Far Field**: The pattern simplifies and eventually resembles that of a single source with modified amplitude

This transition from near to far field is important in many applications, such as antenna arrays and acoustic systems.

### 4. Wavelength Relationship

The spacing between nodal lines is directly related to the wavelength:
- Shorter wavelengths produce more densely packed interference patterns
- Changing the wavelength scales the pattern spatially without changing its fundamental structure
- For a fixed source geometry, the pattern repeats at distances of λ from each source

### 5. Time Evolution

Our animation shows how the interference pattern evolves over time:
- The pattern appears to radiate outward from the sources
- The overall structure of constructive and destructive regions remains fixed in space
- Individual points oscillate between positive and negative displacement
- The animation helps visualize the wave nature of the phenomenon

## Applications and Practical Significance

Understanding water wave interference patterns has applications in various fields:

1. **Wave Engineering**: Designing breakwaters and coastal structures to control wave impact
2. **Acoustics**: Designing speaker arrays for directional sound propagation
3. **Electromagnetic Waves**: Antenna array design for directional transmission and reception
4. **Optical Systems**: Holography, interferometry, and diffraction gratings
5. **Quantum Mechanics**: Understanding electron and matter wave interference
6. **Seismology**: Analyzing seismic wave patterns for geological study

## Conclusions

This analysis demonstrates the rich and complex behavior that emerges when multiple coherent wave sources interact. The resulting interference patterns reveal fundamental properties of waves and the principle of superposition.

Our observations confirm that:
1. The principle of superposition accurately predicts the complex patterns formed by overlapping waves
2. Geometric arrangement of sources directly influences the symmetry and structure of interference patterns
3. As the number of sources increases, the interference patterns become more complex while maintaining the underlying symmetry of the source arrangement

The visualization tools we've developed allow for intuitive understanding of these complex wave phenomena, making abstract concepts tangible and accessible. The ability to manipulate parameters such as wavelength, amplitude, and source geometry provides a powerful framework for exploring wave behavior in various contexts.

These findings highlight the universal nature of wave interference, demonstrating principles that apply across different physical domains—from water waves to light, sound, and quantum mechanical waves.
