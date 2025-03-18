# Investigating the Range as a Function of the Angle of Projection

## Motivation
Projectile motion, while seemingly simple, offers a rich playground for exploring fundamental principles of physics. The problem is straightforward: analyze how the range of a projectile depends on its angle of projection. Yet, beneath this simplicity lies a complex and versatile framework. The equations governing projectile motion involve both linear and quadratic relationships, making them accessible yet deeply insightful.

What makes this topic particularly compelling is the number of free parameters involved in these equations, such as initial velocity, gravitational acceleration, and launch height. These parameters give rise to a diverse set of solutions that can describe a wide array of real-world phenomena, from the arc of a soccer ball to the trajectory of a rocket.

## 1. Theoretical Foundation
### Governing Equations
Projectile motion follows Newton’s second law, and we assume motion under constant acceleration due to gravity, ignoring air resistance.

- The horizontal motion is governed by:
  $$ x = v_0 \cos(\theta) t $$
  
- The vertical motion follows:
  $$ y = v_0 \sin(\theta) t - \frac{1}{2} g t^2 $$
  
Solving for the time of flight when the projectile returns to the ground ($y=0$):

$$ t_f = \frac{2 v_0 \sin(\theta)}{g} $$

The range, which is the horizontal distance traveled, is given by:

$$ R = v_0 \cos(\theta) t_f = \frac{v_0^2 \sin(2\theta)}{g} $$

### Family of Solutions
- The range is maximized when $ \theta = 45^\circ $, as $ \sin(2\theta) $ reaches its peak at this angle.
- Different values of $ v_0 $ and $ g $ shift the entire curve up or down, affecting the overall range.

## 2. Analysis of the Range
- The function $ R(\theta) = \frac{v_0^2 \sin(2\theta)}{g} $ follows a sinusoidal form, reaching its peak at 45 degrees.
- Increasing $ v_0 $ increases the range quadratically.
- A higher gravitational acceleration $ g $ decreases the range.
- If the projectile is launched from a height $ h $, the range expression becomes more complex:
  $$ R = \frac{v_0 \cos(\theta)}{g} \left( v_0 \sin(\theta) + \sqrt{(v_0 \sin(\theta))^2 + 2 g h} \right) $$

## 3. Practical Applications
- **Sports**: Understanding optimal angles for long jumps, soccer kicks, or basketball shots.
- **Engineering**: Ballistics and missile trajectory calculations.
- **Astrophysics**: Studying celestial bodies’ motion in the absence of air resistance.

## 4. Implementation
Below is a Python script to simulate and visualize the range as a function of the launch angle.

```python
import numpy as np
import matplotlib.pyplot as plt

def projectile_range(v0, g):
    angles = np.linspace(0, 90, 100)  # Angles in degrees
    radians = np.radians(angles)  # Convert to radians
    ranges = (v0**2 * np.sin(2 * radians)) / g  # Compute range
    
    plt.figure(figsize=(8, 5))
    plt.plot(angles, ranges, label=f'Initial Velocity = {v0} m/s')
    plt.xlabel('Angle of Projection (degrees)')
    plt.ylabel('Range (m)')
    plt.title('Projectile Range as a Function of Angle')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
projectile_range(v0=20, g=9.81)
```
![alt text](Unknown.png)
## 5. Discussion on Model Limitations
- The model assumes no air resistance, which is unrealistic for real-world projectiles.
- Wind and drag force significantly alter projectile motion.
- For high-speed objects, Coriolis effects (due to Earth's rotation) might need to be considered.
- Uneven terrain or varying gravitational acceleration can affect actual projectile behavior.

## 6. Conclusion
This study highlights the interplay between angle, velocity, and gravity in determining a projectile’s range. The insights gained are applicable across sports, engineering, and even astrophysics. Future work can involve adding air resistance to the model for a more realistic simulation.
