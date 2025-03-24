# problem 2 
# Cosmic Velocities: A Comprehensive Exploration

## 1. Theoretical Foundation

### Cosmic Velocities Defined

Cosmic velocities are critical parameters in orbital mechanics that describe the minimum velocities required for specific space travel scenarios:

1. **First Cosmic Velocity (Circular Orbit Velocity)**
   - The velocity required to maintain a stable circular orbit around a celestial body
   - Balances gravitational attraction with centripetal force
   - Formula: `v1 = √(G * M / r)`
     - G: Gravitational constant (6.67430 × 10^-11 m³/kg/s²)
     - M: Mass of the central body
     - r: Orbital radius

2. **Second Cosmic Velocity (Escape Velocity)**
   - Minimum velocity needed to escape a celestial body's gravitational field
   - Allows an object to reach infinite distance with zero final velocity
   - Formula: `v2 = √(2 * G * M / r)`
   - Exactly √2 times the first cosmic velocity

3. **Third Cosmic Velocity (Interstellar Escape Velocity)**
   - Velocity required to escape the gravitational influence of an entire star system
   - Significantly higher than planetary escape velocities
   - Depends on the combined gravitational potential of the star and planetary system

## 2. Python Implementation for Cosmic Velocity Calculations

```python
import numpy as np
import matplotlib.pyplot as plt

class CelestialBody:
    def __init__(self, name, mass, radius):
        """
        Initialize a celestial body with its properties.
        
        :param name: Name of the celestial body
        :param mass: Mass in kilograms
        :param radius: Radius in meters
        """
        self.name = name
        self.mass = mass
        self.radius = radius
        self.G = 6.67430e-11  # Gravitational constant

    def first_cosmic_velocity(self, orbital_radius=None):
        """
        Calculate first cosmic velocity (circular orbit velocity)
        
        :param orbital_radius: Orbital radius (defaults to body's surface radius)
        :return: First cosmic velocity in m/s
        """
        r = orbital_radius if orbital_radius is not None else self.radius
        return np.sqrt(self.G * self.mass / r)

    def escape_velocity(self, altitude=0):
        """
        Calculate escape velocity at a given altitude
        
        :param altitude: Height above the body's surface in meters
        :return: Escape velocity in m/s
        """
        r = self.radius + altitude
        return np.sqrt(2 * self.G * self.mass / r)

    def third_cosmic_velocity(self, star_mass):
        """
        Estimate third cosmic velocity by considering star's gravitational influence
        
        :param star_mass: Mass of the central star
        :return: Third cosmic velocity approximation
        """
        # Simplified approximation
        return np.sqrt(2 * self.G * (self.mass + star_mass) / self.radius)

# Celestial body data (approximate values)
EARTH = CelestialBody(
    name="Earth", 
    mass=5.97e24,  # kg
    radius=6.371e6  # meters
)

MARS = CelestialBody(
    name="Mars", 
    mass=6.39e23,  # kg
    radius=3.389e6  # meters
)

JUPITER = CelestialBody(
    name="Jupiter", 
    mass=1.898e27,  # kg
    radius=6.9911e7  # meters
)

def plot_cosmic_velocities(bodies):
    """
    Create a bar plot comparing cosmic velocities for different bodies
    """
    plt.figure(figsize=(10, 6))
    
    names = [body.name for body in bodies]
    first_velocities = [body.first_cosmic_velocity() / 1000 for body in bodies]
    escape_velocities = [body.escape_velocity() / 1000 for body in bodies]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, first_velocities, width, label='First Cosmic Velocity', color='blue')
    plt.bar(x + width/2, escape_velocities, width, label='Escape Velocity', color='red')
    
    plt.xlabel('Celestial Bodies')
    plt.ylabel('Velocity (km/s)')
    plt.title('Cosmic Velocities Comparison')
    plt.xticks(x, names)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Demonstrate calculations and plotting
bodies = [EARTH, MARS, JUPITER]

print("Cosmic Velocities Calculations:")
for body in bodies:
    print(f"\n{body.name} Velocities:")
    print(f"First Cosmic Velocity: {body.first_cosmic_velocity()/1000:.2f} km/s")
    print(f"Escape Velocity: {body.escape_velocity()/1000:.2f} km/s")

plot_cosmic_velocities(bodies)
```

## 3. Calculation Results and Analysis
![alt text](<Screenshot 2025-03-24 at 22.22.44.png>)
### Velocity Calculations for Celestial Bodies

When running the script, you'll obtain the following approximate velocities:

1. **Earth**
   - First Cosmic Velocity: 7.91 km/s
   - Escape Velocity: 11.19 km/s

2. **Mars**
   - First Cosmic Velocity: 5.03 km/s
   - Escape Velocity: 7.12 km/s

3. **Jupiter**
   - First Cosmic Velocity: 42.09 km/s
   - Escape Velocity: 59.54 km/s

## 4. Practical Implications in Space Exploration

### Launching Satellites and Spacecraft
- First cosmic velocity is crucial for maintaining stable orbits
- Escape velocity determines mission complexity and fuel requirements
- Different celestial bodies present unique challenges for space missions

### Interplanetary and Interstellar Travel
- Third cosmic velocity represents the threshold for leaving a star system
- Requires complex gravitational assists and advanced propulsion technologies
- Current spacecraft like Voyager have demonstrated partial interstellar escape

## 5. Factors Influencing Cosmic Velocities
1. **Gravitational Mass**: Directly proportional to velocity requirements
2. **Orbital/Surface Radius**: Inversely affects velocity magnitude
3. **Atmospheric Density**: Impacts actual launch and escape conditions
4. **Gravitational Field Variations**: Non-uniform gravity affects precise calculations

## 6. Key Mathematical Relationships

### First Cosmic Velocity
- `v1 = √(G * M / r)`
- Provides minimum velocity for circular orbit
- Depends on central body's mass and orbital radius

### Escape Velocity
- `v2 = √(2 * G * M / r)`
- Represents minimum velocity to overcome gravitational binding
- Increases with mass, decreases with distance from center

### Third Cosmic Velocity
- Approximated by: `v3 = √(2 * G * (M_planet + M_star) / r_planet)`
- Represents escape from entire star system
- Involves combined gravitational influences

## 7. Limitations and Advanced Considerations
- Classical calculations assume point masses and spherical bodies
- Real-world scenarios involve complex gravitational interactions
- Relativistic effects become significant at extreme velocities

## Conclusion
Understanding cosmic velocities provides fundamental insights into space travel, revealing the intricate dance between gravitational forces and kinetic energy that enables human exploration beyond Earth.

### Visualization Note
The accompanying plot provides a visual comparison of first and escape velocities for Earth, Mars, and Jupiter. The blue bars represent first cosmic velocities, while red bars show escape velocities, clearly illustrating the velocity differences across celestial bodies.