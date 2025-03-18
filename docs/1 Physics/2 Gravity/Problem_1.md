# Problem 1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_sun = 1.989e30  # Mass of the Sun (kg)

# Function to create circular orbit coordinates
def circular_orbit(radius, period, num_points=100):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y

# Define planets data (radius in AU, period in years)
planets = {
    'Mercury': (0.39, 0.24),
    'Venus': (0.72, 0.62),
    'Earth': (1.0, 1.0),
    'Mars': (1.52, 1.88),
    'Jupiter': (5.20, 11.86)
}

# Convert AU to meters and years to seconds for calculations
AU = 1.496e11  # 1 AU in meters
year = 365.25 * 24 * 3600  # 1 year in seconds

# Calculate T^2/r^3 for each planet
t2_r3_values = {}
for planet, (r_au, t_yr) in planets.items():
    r = r_au * AU
    t = t_yr * year
    t2_r3 = (t**2) / (r**3)
    t2_r3_values[planet] = t2_r3

# Calculate theoretical value of 4π²/(G*M_sun)
theoretical = 4 * np.pi**2 / (G * M_sun)

# Plot orbits
plt.figure(figsize=(10, 10))
plt.title('Planetary Orbits in the Solar System (Not to Scale)', fontsize=14)

colors = ['gray', 'orange', 'blue', 'red', 'brown']
for i, (planet, (radius, _)) in enumerate(planets.items()):
    x, y = circular_orbit(radius, planets[planet][1])
    plt.plot(x, y, label=planet, color=colors[i])

plt.plot(0, 0, 'yo', markersize=15, label='Sun')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.axis('equal')
plt.xlabel('Distance (AU)', fontsize=12)
plt.ylabel('Distance (AU)', fontsize=12)
plt.savefig('solar_system_orbits.png', dpi=300, bbox_inches='tight')

# Plot T^2 vs r^3
plt.figure(figsize=(10, 6))
plt.title("Kepler's Third Law: T² vs r³", fontsize=14)

r3_values = []
t2_values = []
for planet, (r, t) in planets.items():
    r3 = r**3
    t2 = t**2
    r3_values.append(r3)
    t2_values.append(t2)
    plt.scatter(r3, t2, s=100, label=planet)

# Add best fit line
plt.plot(np.array(r3_values), np.array(r3_values), 'k--', alpha=0.7, label='T² = r³')
plt.xlabel('r³ (AU³)', fontsize=12)
plt.ylabel('T² (years²)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.savefig('kepler_third_law.png', dpi=300, bbox_inches='tight')

# Create an animation of the orbits
def animate_orbits():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Planetary Motion in the Solar System (Time-scaled)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_xlabel('Distance (AU)', fontsize=12)
    ax.set_ylabel('Distance (AU)', fontsize=12)
    ax.plot(0, 0, 'yo', markersize=15)  # Sun
    
    # Create orbit lines
    orbit_lines = []
    planet_dots = []
    for i, (planet, (radius, _)) in enumerate(planets.items()):
        x, y = circular_orbit(radius, planets[planet][1])
        orbit_line, = ax.plot(x, y, alpha=0.3, color=colors[i])
        planet_dot, = ax.plot([], [], 'o', color=colors[i], markersize=10, label=planet)
        orbit_lines.append(orbit_line)
        planet_dots.append(planet_dot)
    
    ax.legend(fontsize=12)
    
    # Set different speeds proportional to actual orbital periods
    speeds = [2*np.pi/period for _, period in planets.values()]
    max_speed = max(speeds)
    speeds = [s/max_speed*0.1 for s in speeds]  # Normalize speeds
    
    def init():
        for dot in planet_dots:
            dot.set_data([], [])
        return planet_dots
    
    def animate(i):
        for j, ((_, (radius, _)), dot, speed) in enumerate(zip(planets.items(), planet_dots, speeds)):
            angle = i * speed
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            dot.set_data(x, y)
        return planet_dots
    
    ani = FuncAnimation(fig, animate, frames=200, init_func=init, blit=True, interval=50)
    return ani
![alt text](download.png)

# Create and display the animation
ani = animate_orbits()
plt.close()  # Close the animation figure to avoid displaying it twice

# Print verification of Kepler's Third Law
print("Verification of Kepler's Third Law:")
print(f"{'Planet':<10} {'T²/r³ (s²/m³)':<20} {'% of theoretical':<15}")
print("-" * 45)
for planet, value in t2_r3_values.items():
    percentage = (value / theoretical) * 100
    print(f"{planet:<10} {value:.6e} {percentage:.2f}%")
print("\nTheoretical value (4π²/GM_sun):", f"{theoretical:.6e}")

# Print the relation in more intuitive units
print("\nIn more intuitive units:")
print("For planets orbiting the Sun: T² (in years) ≈ r³ (in AU)")

# Demonstrate the use of Kepler's Third Law for mass calculation
print("\nUsing Kepler's Third Law to calculate the Sun's mass:")
r_earth = 1.0 * AU  # Earth's orbital radius in meters
t_earth = 1.0 * year  # Earth's orbital period in seconds
calculated_mass = 4 * np.pi**2 * r_earth**3 / (G * t_earth**2)
print(f"Calculated Sun's mass: {calculated_mass:.3e} kg")
print(f"Actual Sun's mass:     {M_sun:.3e} kg")
print(f"Difference: {abs(calculated_mass - M_sun)/M_sun*100:.4f}%")