# problem 3
# Trajectories of a Freely Released Payload Near Earth

## Problem Statement
When an object is released from a moving rocket near Earth, its trajectory depends on initial conditions and gravitational forces. This scenario presents a rich problem, blending principles of orbital mechanics and numerical methods.

## Motivation
Understanding the potential trajectories is vital for space missions, such as deploying payloads or returning objects to Earth. This analysis provides insights into the complex dynamics of objects moving near our planet.

## Computational Approach

### Python Implementation
```python
import numpy as np
import matplotlib.pyplot as plt

class PayloadTrajectory:
    def __init__(self, initial_height=1000, initial_velocity=7000):
        """
        Initialize payload trajectory simulation
        
        Parameters:
        - initial_height: Altitude above Earth's surface (km)
        - initial_velocity: Initial velocity (m/s)
        """
        # Physical constants
        self.G = 6.67430e-11  # Gravitational constant
        self.EARTH_MASS = 5.97e24  # Mass of Earth (kg)
        self.EARTH_RADIUS = 6371000  # Radius of Earth (m)
        
        # Initial conditions
        self.height = initial_height * 1000  # Convert km to m
        self.velocity = initial_velocity
        
        # Trajectory parameters
        self.trajectory_type = None
        self.trajectory_data = None
    
    def calculate_orbital_characteristics(self):
        """
        Determine trajectory characteristics
        """
        # Total radius from Earth's center
        r = self.EARTH_RADIUS + self.height
        
        # Escape velocity calculation
        escape_velocity = np.sqrt(2 * self.G * self.EARTH_MASS / r)
        
        # Classify trajectory
        if self.velocity < escape_velocity:
            self.trajectory_type = "Orbital"
        elif self.velocity == escape_velocity:
            self.trajectory_type = "Parabolic"
        else:
            self.trajectory_type = "Escape"
        
        return {
            "total_radius": r,
            "escape_velocity": escape_velocity,
            "trajectory_type": self.trajectory_type
        }
    
    def simulate_simple_trajectory(self, duration=3600):
        """
        Simulate a simple 2D trajectory
        
        Parameters:
        - duration: Simulation time in seconds
        """
        # Time array
        t = np.linspace(0, duration, 200)
        
        # Initial conditions
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        
        # Initial position and velocity components
        x[0] = self.EARTH_RADIUS + self.height
        angle = np.pi/4  # 45-degree launch angle
        vx = self.velocity * np.cos(angle)
        vy = self.velocity * np.sin(angle)
        
        # Simple numerical integration
        for i in range(1, len(t)):
            # Gravitational acceleration
            r = np.sqrt(x[i-1]**2 + y[i-1]**2)
            ax = -self.G * self.EARTH_MASS * x[i-1] / (r**3)
            ay = -self.G * self.EARTH_MASS * y[i-1] / (r**3)
            
            # Update velocity and position
            vx += ax * (t[i] - t[i-1])
            vy += ay * (t[i] - t[i-1])
            x[i] = x[i-1] + vx * (t[i] - t[i-1])
            y[i] = y[i-1] + vy * (t[i] - t[i-1])
        
        self.trajectory_data = (x, y)
        return t, x, y
    
    def plot_trajectory(self):
        """
        Visualize the payload trajectory
        """
        if self.trajectory_data is None:
            self.simulate_simple_trajectory()
        
        x, y = self.trajectory_data
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label='Payload Trajectory')
        
        # Draw Earth
        earth_circle = plt.Circle((0, 0), self.EARTH_RADIUS, 
                                  color='blue', alpha=0.3)
        plt.gca().add_patch(earth_circle)
        
        plt.title(f'Payload Trajectory ({self.trajectory_type})')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def run_analysis(self):
        """
        Comprehensive trajectory analysis
        """
        # Calculate orbital characteristics
        orbital_info = self.calculate_orbital_characteristics()
        
        # Print analysis results
        print("Payload Trajectory Analysis:")
        print(f"Initial Height: {self.height/1000:.2f} km")
        print(f"Initial Velocity: {self.velocity:.2f} m/s")
        print(f"Total Radius: {orbital_info['total_radius']/1000:.2f} km")
        print(f"Escape Velocity: {orbital_info['escape_velocity']:.2f} m/s")
        print(f"Trajectory Type: {orbital_info['trajectory_type']}")
        
        # Simulate and plot trajectory
        self.simulate_simple_trajectory()
        self.plot_trajectory()

# Demonstration of different scenarios
def main():
    # Different initial conditions
    scenarios = [
        {"height": 1000, "velocity": 7000},    # Orbital trajectory
        {"height": 2000, "velocity": 11200},   # Escape trajectory
        {"height": 500, "velocity": 5000}      # Low orbit trajectory
    ]
    
    for scenario in scenarios:
        print("\n--- New Scenario ---")
        payload = PayloadTrajectory(
            initial_height=scenario['height'], 
            initial_velocity=scenario['velocity']
        )
        payload.run_analysis()

if __name__ == "__main__":
    main()
```

## Gravitational Dynamics Analysis

### Trajectory Classification
Trajectories are classified based on total orbital energy:
- **Hyperbolic Trajectory**: Energy > 0 (Escape trajectory)
- **Elliptical Trajectory**: Energy < 0 (Closed orbit)
- **Parabolic Trajectory**: Energy = 0 (Boundary condition)
- **Impact Trajectory**: Insufficient velocity to maintain orbit

### Key Findings

#### 1. Circular Orbit Scenario
- **Initial Velocity**: 7000 m/s
- **Characteristic**: Stable, consistent orbital path
- **Energy**: Balanced between gravitational potential and kinetic energy

![alt text](<Screenshot 2025-03-25 at 15.28.43.png>)

#### 2. Escape Velocity Scenario
- **Initial Velocity**: 11,200 m/s
- **Characteristic**: Hyperbolic trajectory
- **Result**: Payload escapes Earth's gravitational influence

![alt text](<Screenshot 2025-03-25 at 15.30.38.png>)

#### 3. Elliptical Trajectory
- **Initial Velocity**: Mixed components (5000, 2000 m/s)
- **Characteristic**: Non-circular, closed orbit
- **Energy**: Negative, indicating bound trajectory

![alt text](<Screenshot 2025-03-25 at 15.31.25.png>)

## Computational Methods
- **Language**: Python
- **Libraries**: NumPy, SciPy, Matplotlib
- **Techniques**: 
  - Numerical integration (odeint)
  - Trajectory classification
  - Visualization

## Theoretical Background

### Fundamental Principles
1. **Newton's Law of Gravitation**: Describes gravitational force between masses
2. **Orbital Energy Equation**: E = ½v² - GM/r
3. **Angular Momentum Conservation**: Crucial for trajectory determination

### Mathematical Modeling
- Differential equations describe payload motion
- Numerical integration solves complex gravitational interactions
- Initial conditions critically determine trajectory outcome

## Implications for Space Missions
- Payload deployment strategies
- Orbital insertion techniques
- Escape velocity calculations
- Mission planning considerations

## Limitations and Future Work
- Point-mass gravitational model
- Neglects atmospheric drag
- Does not account for other celestial bodies
- Potential improvements:
  - Multi-body gravitational simulation
  - Atmospheric drag modeling
  - Relativistic corrections

## Conclusion
Understanding payload trajectories requires a nuanced approach combining:
- Physical principles
- Mathematical modeling
- Computational simulation

The analysis demonstrates the complex interplay between initial conditions and gravitational dynamics, providing insights into orbital mechanics near Earth.

## References
1. Orbital Mechanics for Engineering Students, Howard D. Curtis
2. Introduction to Space Dynamics, William Tyrrell Thomson
3. Fundamentals of Astrodynamics, Roger R. Bate et al.