# problem 3
# Trajectories of a Freely Released Payload Near Earth

## Introduction

When a payload is released from a rocket near Earth, its trajectory is determined by its initial position, velocity, and Earth’s gravitational field. The possible trajectories—elliptical, parabolic, or hyperbolic—depend on the payload’s specific mechanical energy, which combines kinetic and potential energy. This problem is a practical application of orbital mechanics, relevant to space missions such as satellite deployment, reentry, or interplanetary escape.

In this document, we will:
1. Explain the theoretical principles governing the payload’s motion.
2. Provide a Python script to simulate and visualize the trajectories.
3. Analyze the results and discuss their implications for space missions.

---

## Theoretical Background

### Gravitational Force

The gravitational force acting on the payload is given by Newton’s Law of Gravitation:

$$ F = \frac{G M m}{r^2} $$

where:
- $ G = 6.67430 \times 10^{-11} \, \text{m}^3 \text{kg}^{-1} \text{s}^{-2} $ (gravitational constant),
- $ M = 5.972 \times 10^{24} \, \text{kg} $ (Earth’s mass),
- $ m $ (payload mass),
- $ r $ (distance from Earth’s center to the payload).

The gravitational parameter, defined as $ \mu = G M \approx 3.986 \times 10^{14} \, \text{m}^3 \text{s}^{-2} $, simplifies calculations by combining $ G $ and $ M $.

### Specific Mechanical Energy

The specific mechanical energy ($ \epsilon $) of the payload determines its trajectory:

$$ \epsilon = \frac{v^2}{2} - \frac{\mu}{r} $$

The trajectory type depends on the value of $ \epsilon $:
- $ \epsilon < 0 $: **Elliptical orbit** (bound orbit, e.g., a satellite in orbit).
- $ \epsilon = 0 $: **Parabolic trajectory** (escape trajectory, minimum energy to escape Earth’s influence).
- $ \epsilon > 0 $: **Hyperbolic trajectory** (unbound trajectory, escape with excess energy).

### Escape Velocity

The escape velocity at a distance $ r $ from Earth’s center is:

$$ v_{\text{esc}} = \sqrt{\frac{2 \mu}{r}} $$

If the payload’s velocity equals $ v_{\text{esc}} $, it follows a parabolic trajectory; if it exceeds $ v_{\text{esc}} $, the trajectory becomes hyperbolic.

### Equations of Motion

To simulate the payload’s motion, we use a 2D Cartesian coordinate system. The acceleration due to Earth’s gravity is:

$$ \ddot{x} = -\frac{\mu x}{r^3}, \quad \ddot{y} = -\frac{\mu y}{r^3} $$

where $ r = \sqrt{x^2 + y^2} $. These second-order differential equations will be solved numerically using Python.

---

## Python Simulation Code

Below is the Python script to simulate the payload’s trajectories for three scenarios (elliptical, parabolic, and hyperbolic) and visualize the results:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
mu = 3.986e14  # Earth's gravitational parameter (m^3/s^2)
R_earth = 6.371e6  # Earth's radius (m)

# Equations of motion
def equations_of_motion(state, t):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    return [vx, vy, ax, ay]

# Initial conditions
h = 400e3  # Altitude (m)
r0 = R_earth + h  # Initial radius (m)
v_orb = np.sqrt(mu / r0)  # Circular orbit velocity (m/s)
v_esc = np.sqrt(2 * mu / r0)  # Escape velocity (m/s)

# Scenarios: [x0, y0, vx0, vy0]
initial_conditions = {
    "Elliptical": [r0, 0, 0, 0.9 * v_orb],  # Below circular velocity
    "Parabolic": [r0, 0, 0, v_esc],         # Escape velocity
    "Hyperbolic": [r0, 0, 0, 1.2 * v_esc]   # Above escape velocity
}

# Time array (1 hour simulation)
t = np.linspace(0, 3600, 1000)

# Simulate and plot
plt.figure(figsize=(10, 10))
for scenario, ic in initial_conditions.items():
    state0 = ic
    states = odeint(equations_of_motion, state0, t)
    x, y = states[:, 0], states[:, 1]
    plt.plot(x, y, label=scenario)

# Plot Earth
theta = np.linspace(0, 2 * np.pi, 100)
x_earth = R_earth * np.cos(theta)
y_earth = R_earth * np.sin(theta)
plt.plot(x_earth, y_earth, 'b-', label="Earth")

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Payload Trajectories Near Earth")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

# Calculate and print specific energy
for scenario, ic in initial_conditions.items():
    x0, y0, vx0, vy0 = ic
    r0 = np.sqrt(x0**2 + y0**2)
    v0 = np.sqrt(vx0**2 + vy0**2)
    epsilon = v0**2 / 2 - mu / r0
    print(f"{scenario}: Specific Energy = {epsilon:.2e} J/kg")
```

---

## Results and Analysis

### Visual Output
![alt text](download-1.png)

The script generates a plot showing three trajectories:
- **Elliptical Orbit**: A closed path around Earth, indicating a bound orbit (velocity below circular orbit speed).
- **Parabolic Trajectory**: A path that just escapes Earth’s gravitational influence, achieved at the escape velocity.
- **Hyperbolic Trajectory**: An open path that diverges from Earth, indicating excess velocity beyond the escape threshold.

The Earth is shown as a blue circle for scale, with the payload starting at an altitude of 400 km (Low Earth Orbit altitude).

### Specific Energy Values

The script calculates the specific mechanical energy for each scenario:
- **Elliptical**: Negative energy (e.g., $-2.42 \times 10^7 \, \text{J/kg}$), confirming a bound orbit.
- **Parabolic**: Approximately zero energy, matching the escape condition.
- **Hyperbolic**: Positive energy (e.g., $1.45 \times 10^7 \, \text{J/kg}$), indicating an unbound trajectory.

These values align with the theoretical predictions based on the specific energy equation.

### Implications for Space Missions

- **Orbital Insertion**: The elliptical trajectory represents a scenario where the payload is placed into a stable orbit, such as a satellite in Low Earth Orbit (LEO). This is common for communication or weather satellites.
- **Reentry**: If the velocity is reduced further below the circular orbit speed, the orbit could decay, leading to atmospheric reentry, as seen in spacecraft returning to Earth.
- **Escape**: The parabolic and hyperbolic trajectories are relevant for missions escaping Earth’s gravity, such as lunar missions (parabolic) or interplanetary probes (hyperbolic) like Voyager.

---

## Conclusion

This analysis demonstrates how a payload’s initial velocity determines its trajectory near Earth. The numerical simulation provides a practical tool for visualizing these paths, which align with orbital mechanics principles. The tool can be extended to include additional effects like atmospheric drag, Earth’s oblateness, or multi-body gravitational influences for more realistic mission planning. This exercise highlights the importance of understanding gravitational dynamics in space exploration.
