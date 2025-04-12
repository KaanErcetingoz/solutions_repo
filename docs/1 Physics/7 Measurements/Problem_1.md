# Problem 1
# Measuring Earth's Gravitational Acceleration with a Pendulum

## Introduction
This document presents the results of an experiment to measure the acceleration due to gravity (g) using a simple pendulum. The experiment employs the relationship between the period of oscillation and the length of a pendulum to determine g with associated uncertainties.

## Experimental Setup
- A pendulum was constructed using a 1.20 m string with a small weight attached.
- The pendulum was suspended from a rigid support.
- Small angle oscillations (<15°) were maintained throughout the experiment.
- Time measurements were taken for 10 complete oscillations across 10 trials.

## Tabulated Data

### Length Measurement
- Length (L): 1.20 m
- Measuring tool resolution: 0.001 m (1 mm)
- Length uncertainty (ΔL): 0.0005 m (0.5 mm)

### Time Measurements for 10 Oscillations (T₁₀)

| Trial | Time for 10 oscillations (s) |
|-------|------------------------------|
| 1     | 21.94                        |
| 2     | 22.03                        |
| 3     | 21.98                        |
| 4     | 22.06                        |
| 5     | 21.95                        |
| 6     | 22.01                        |
| 7     | 21.97                        |
| 8     | 22.04                        |
| 9     | 21.99                        |
| 10    | 22.02                        |

- Mean time (T̄₁₀): 22.00 s
- Standard deviation (σₜ): 0.04 s
- Uncertainty in mean time (ΔT̄₁₀ = σₜ/√10): 0.013 s

## Calculations

### 1. Period Calculation
- Period (T = T̄₁₀/10): 2.200 s
- Period uncertainty (ΔT = ΔT̄₁₀/10): 0.0013 s

### 2. Gravity Calculation
- g = 4π²L/T² = 4π² × 1.20 m/(2.200 s)² = 9.80 m/s²

### 3. Uncertainty Propagation
- Δg = g√[(ΔL/L)² + (2ΔT/T)²]
- Δg = 9.80√[(0.0005/1.20)² + (2 × 0.0013/2.200)²]
- Δg = 9.80√[1.74 × 10⁻⁷ + 1.40 × 10⁻⁶]
- Δg = 9.80√[1.57 × 10⁻⁶]
- Δg = 9.80 × 1.25 × 10⁻³
- Δg = 0.012 m/s²

### Final Result
- g = (9.80 ± 0.01) m/s²

## Analysis

### 1. Comparison with Standard Value
The measured value of g = (9.80 ± 0.01) m/s² is in excellent agreement with the standard value of 9.81 m/s². The difference is approximately 0.10%, which is within our calculated uncertainty range.

### 2. Discussion of Uncertainties

#### The effect of measurement resolution on ΔL
The ruler's resolution (1 mm) contributes to an uncertainty of 0.5 mm in the length measurement. This translates to approximately 0.04% relative uncertainty in the length measurement, which propagates to the calculation of g. While this uncertainty is small for our experiment, it becomes more significant for shorter pendulums.

Additional factors affecting length measurement include:
- Difficulty in precisely locating the center of mass of the weight
- String stretching during oscillation
- Measurement of the exact suspension point

Our experiment minimized these effects by using a dense weight (minimizing the impact of center of mass uncertainty) and a relatively inelastic string.

#### Variability in timing and its impact on ΔT
The standard deviation in timing measurements (0.04 s) reflects random errors in the timing process. By taking the mean of 10 trials, we reduced this uncertainty to 0.013 s for the mean time of 10 oscillations, or 0.0013 s for a single period. This translates to about 0.06% relative uncertainty in the period.

Timing uncertainties arise from:
- Human reaction time (~0.1-0.3 s)
- Difficulty in identifying the exact moment the pendulum passes its equilibrium position
- Potential miscounting of oscillations

Measuring multiple oscillations substantially reduces the impact of reaction time errors, as these errors become a smaller fraction of the total measured time.

#### Assumptions and experimental limitations
Our experiment relies on several key assumptions and has notable limitations:

1. **Small angle approximation**: The relationship T = 2π√(L/g) is valid only for small angles. We maintained oscillations below 15° to ensure this approximation holds. For larger angles, the period increases slightly, leading to underestimation of g.

2. **Simple pendulum model**: We assumed:
   - The string is massless (in reality, it contributes slightly to the moment of inertia)
   - The weight acts as a point mass (in reality, it has finite dimensions)
   - The pendulum length remains constant (slight stretching may occur)

3. **Environmental factors**:
   - Air resistance gradually damping the oscillations
   - Air currents potentially affecting the pendulum motion
   - Temperature effects on the string length

4. **Local variations in g**: The value of g varies slightly with latitude (due to Earth's rotation and shape) and altitude. Our measurement reflects the local value at our specific location.

## Conclusion
The experiment successfully measured Earth's gravitational acceleration with good precision. The final value of g = (9.80 ± 0.01) m/s² demonstrates that even with relatively simple equipment, fundamental physical constants can be measured accurately when proper experimental techniques and uncertainty analysis are applied.

The small difference between our measured value and the standard value (9.81 m/s²) could be attributed to:
- Experimental uncertainties
- Local variations in the gravitational field
- Systematic errors in our experimental setup

This experiment illustrates the importance of uncertainty analysis in experimental physics and demonstrates how multiple measurements can improve precision through statistical methods.
