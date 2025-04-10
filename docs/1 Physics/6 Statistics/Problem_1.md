# Problem 1

# Exploring the Central Limit Theorem through Simulations

## Introduction

The Central Limit Theorem (CLT) is a fundamental concept in probability theory and statistics that describes the behavior of the sampling distribution of the mean for sufficiently large sample sizes. According to the CLT, regardless of the original population distribution, the sampling distribution of the sample mean will approach a normal distribution as the sample size increases. This report presents a comprehensive exploration of the CLT through computational simulations using Python.

## Theoretical Background

The Central Limit Theorem states that if we take sufficiently large random samples from any population with a finite mean μ and variance σ², then the sampling distribution of the sample mean will be approximately normally distributed with mean μ and standard deviation σ/√n, where n is the sample size.

Mathematically, if X₁, X₂, ..., Xₙ are independent and identically distributed random variables with mean μ and variance σ², then as n approaches infinity:

$$\bar{X}_n \sim N(\mu, \frac{\sigma^2}{n})$$

Where $\bar{X}_n$ is the sample mean.

## Simulation Methodology

Our simulation approach explores how the CLT manifests across different probability distributions and sample sizes:

1. **Population Distributions**: We selected several distinct distributions to demonstrate the universality of the CLT:
   - Uniform distribution (U[0,10])
   - Exponential distribution (λ=0.5)
   - Binomial distribution (n=20, p=0.3)
   - Chi-square distribution (df=3)

2. **Sampling Process**:
   - For each distribution, we generated a large population (1,000,000 data points)
   - We then drew random samples of various sizes (n=5, 10, 30, 50, 100)
   - For each sample size, we repeated the sampling process 10,000 times
   - We calculated and stored the mean of each sample to create the sampling distribution

3. **Analysis Methods**:
   - Visualization using histograms and density plots
   - Q-Q plots to assess normality
   - Kolmogorov-Smirnov tests to quantify convergence to normality
   - Comparison of observed vs. theoretical standard errors

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set the random seed for reproducibility
np.random.seed(42)

# Set global figure parameters
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
plt.style.use('seaborn-v0_8-whitegrid')

# Sample sizes to investigate
sample_sizes = [5, 10, 30, 50]

# Number of times to sample for each sample size
num_samples = 10000

def simulate_sampling_distribution(distribution_name, population_generator, population_params, 
                                  theoretical_mean, theoretical_std, xlim=None):
    """
    Simulates sampling distributions for different sample sizes from a given population distribution.
    
    Parameters:
    - distribution_name: String name of the distribution
    - population_generator: Function to generate population samples
    - population_params: Parameters for the population generator
    - theoretical_mean: Theoretical mean of the population
    - theoretical_std: Theoretical standard deviation of the population
    - xlim: Optional tuple for x-axis limits on the plots
    """
    # Generate a large dataset to represent the population
    population_size = 1000000
    population = population_generator(*population_params, size=population_size)
    
    # Create figure with subplots
    fig, axs = plt.subplots(len(sample_sizes), 2, figsize=(15, 4*len(sample_sizes)))
    fig.suptitle(f'Central Limit Theorem Simulation for {distribution_name} Distribution', fontsize=16)
    
    # Compute actual population statistics
    actual_mean = population.mean()
    actual_std = population.std()
    
    print(f"\n{distribution_name} Distribution:")
    print(f"Theoretical Mean: {theoretical_mean}, Actual Mean: {actual_mean}")
    print(f"Theoretical Std: {theoretical_std}, Actual Std: {actual_std}")
    
    # Plot original population distribution in the first row
    axs[0, 0].hist(population[:10000], bins=50, density=True, alpha=0.7)
    axs[0, 0].set_title(f'Population Distribution (showing 10,000 samples)')
    min_val, max_val = np.min(population[:10000]), np.max(population[:10000])
    x_range = np.linspace(min_val, max_val, 1000)
    
    # If the population is uniform, overlay the PDF
    if distribution_name == "Uniform":
        a, b = population_params
        pdf = np.ones_like(x_range) / (b - a) * ((x_range >= a) & (x_range <= b))
        axs[0, 0].plot(x_range, pdf, 'r-', lw=2, label='PDF')
        axs[0, 0].legend()
    
    if xlim:
        axs[0, 0].set_xlim(xlim)
    
    # For each sample size
    for i, n in enumerate(sample_sizes):
        # Sample means storage
        sample_means = np.zeros(num_samples)
        
        # Perform repeated sampling
        for j in range(num_samples):
            sample = np.random.choice(population, size=n)
            sample_means[j] = np.mean(sample)
        
        # Calculate statistics for the sampling distribution
        mean_of_means = np.mean(sample_means)
        std_of_means = np.std(sample_means)
        expected_std = theoretical_std / np.sqrt(n)
        
        print(f"Sample Size {n}:")
        print(f"  Mean of Sample Means: {mean_of_means:.4f} (Expected: {theoretical_mean:.4f})")
        print(f"  Std of Sample Means: {std_of_means:.4f} (Expected: {expected_std:.4f})")
        
        # Plot the sampling distribution
        ax = axs[i, 1]
        sns.histplot(sample_means, kde=True, stat="density", ax=ax)
        
        # Overlay a normal distribution with the theoretical parameters
        x = np.linspace(np.min(sample_means), np.max(sample_means), 1000)
        y = stats.norm.pdf(x, theoretical_mean, theoretical_std / np.sqrt(n))
        ax.plot(x, y, 'r-', lw=2, label=f'Normal PDF\nμ={theoretical_mean:.2f}, σ={expected_std:.4f}')
        
        # Create a sample visualization in the left column
        axs[i, 0].hist(np.random.choice(population, size=n), bins=20, alpha=0.7)
        axs[i, 0].set_title(f'Example Sample (n={n})')
        if xlim:
            axs[i, 0].set_xlim(xlim)
        
        # Set titles and labels
        ax.set_title(f'Sampling Distribution of Mean (n={n}, samples={num_samples})')
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Density')
        ax.legend()
        
        # Set reasonable x-limits based on the theoretical mean and standard deviation
        if xlim:
            ax.set_xlim(xlim)
        else:
            margin = 4 * expected_std
            ax.set_xlim(theoretical_mean - margin, theoretical_mean + margin)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    return population, sample_means

# 1. Uniform Distribution Simulation
print("\n==== Uniform Distribution ====")
a, b = 0, 10  # Parameters for uniform distribution
theoretical_mean = (a + b) / 2  # Mean of uniform distribution is (a + b) / 2
theoretical_var = (b - a)**2 / 12  # Variance of uniform distribution is (b-a)^2/12
theoretical_std = np.sqrt(theoretical_var)

uniform_population, uniform_sample_means = simulate_sampling_distribution(
    "Uniform",
    np.random.uniform,
    (a, b),
    theoretical_mean,
    theoretical_std,
    xlim=(a-1, b+1)
)

# 2. Exponential Distribution Simulation
print("\n==== Exponential Distribution ====")
rate = 0.5  # Rate parameter for exponential distribution
theoretical_mean = 1 / rate  # Mean of exponential distribution is 1/λ
theoretical_var = 1 / (rate**2)  # Variance of exponential distribution is 1/λ²
theoretical_std = np.sqrt(theoretical_var)

exponential_population, exponential_sample_means = simulate_sampling_distribution(
    "Exponential",
    np.random.exponential,
    (1/rate,),
    theoretical_mean,
    theoretical_std,
    xlim=(0, 10)
)

# 3. Binomial Distribution Simulation
print("\n==== Binomial Distribution ====")
n_trials = 20
p_success = 0.3
theoretical_mean = n_trials * p_success  # Mean of binomial distribution is n*p
theoretical_var = n_trials * p_success * (1 - p_success)  # Variance is n*p*(1-p)
theoretical_std = np.sqrt(theoretical_var)

binomial_population, binomial_sample_means = simulate_sampling_distribution(
    "Binomial",
    np.random.binomial,
    (n_trials, p_success),
    theoretical_mean,
    theoretical_std,
    xlim=(0, n_trials)
)

# 4. Right-skewed Distribution (Chi-Square)
print("\n==== Chi-Square Distribution ====")
df = 3  # Degrees of freedom
theoretical_mean = df
theoretical_std = np.sqrt(2 * df)

chisq_population, chisq_sample_means = simulate_sampling_distribution(
    "Chi-Square",
    np.random.chisquare,
    (df,),
    theoretical_mean,
    theoretical_std,
    xlim=(0, 15)
)

# Function to create comparative plots across different sample sizes
def compare_qq_plots(distribution_name, sample_means_dict):
    """Create Q-Q plots to assess normality of the sampling distributions."""
    fig, axs = plt.subplots(1, len(sample_sizes), figsize=(16, 4))
    fig.suptitle(f'Q-Q Plots for {distribution_name} Sampling Distributions', fontsize=16)
    
    for i, n in enumerate(sample_sizes):
        sample_means = sample_means_dict[n]
        stats.probplot(sample_means, dist="norm", plot=axs[i])
        axs[i].set_title(f'Sample Size n={n}')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

# Function to combine sampling distributions into a single plot for comparison
def compare_distributions(distribution_name, population_generator, population_params,
                         theoretical_mean, theoretical_std):
    """Compare sampling distributions for different sample sizes in a single plot."""
    population_size = 1000000
    population = population_generator(*population_params, size=population_size)
    
    plt.figure(figsize=(12, 8))
    
    # For each sample size
    sample_means_dict = {}
    for n in sample_sizes:
        # Sample means storage
        sample_means = np.zeros(num_samples)
        
        # Perform repeated sampling
        for j in range(num_samples):
            sample = np.random.choice(population, size=n)
            sample_means[j] = np.mean(sample)
        
        sample_means_dict[n] = sample_means
        
        # Plot the density of the sampling distribution
        sns.kdeplot(sample_means, label=f'n={n}')
    
    # Add a line for the theoretical normal distribution
    x = np.linspace(theoretical_mean - 4*theoretical_std/np.sqrt(min(sample_sizes)),
                   theoretical_mean + 4*theoretical_std/np.sqrt(min(sample_sizes)), 1000)
    plt.plot(x, stats.norm.pdf(x, theoretical_mean, theoretical_std/np.sqrt(max(sample_sizes))),
            'r--', lw=2, label=f'Normal (n={max(sample_sizes)})')
    
    plt.title(f'Comparison of Sampling Distributions for {distribution_name}')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Create Q-Q plots
    compare_qq_plots(distribution_name, sample_means_dict)
    
    return sample_means_dict

# Function to investigate convergence of sample mean and variance
def investigate_convergence(distribution_name, population_generator, population_params, 
                           theoretical_mean, theoretical_std):
    """Investigate how quickly the sample mean converges to the population mean."""
    # Generate a large population
    population_size = 1000000
    population = population_generator(*population_params, size=population_size)
    
    # Range of sample sizes to investigate (more detailed)
    detailed_sample_sizes = [2, 3, 5, 10, 15, 20, 30, 50, 100, 200, 500]
    
    # Number of repetitions for each sample size
    repetitions = 1000
    
    # Store results
    mean_errors = []
    std_ratios = []
    
    for n in detailed_sample_sizes:
        sample_means = np.zeros(repetitions)
        sample_stds = np.zeros(repetitions)
        
        for i in range(repetitions):
            sample = np.random.choice(population, size=n)
            sample_means[i] = np.mean(sample)
            sample_stds[i] = np.std(sample, ddof=1)  # Use unbiased estimator
        
        # Calculate mean absolute error for means
        mean_error = np.mean(np.abs(sample_means - theoretical_mean))
        mean_errors.append(mean_error)
        
        # Calculate ratio of std of sample means to theoretical std/sqrt(n)
        expected_std = theoretical_std / np.sqrt(n)
        observed_std = np.std(sample_means)
        std_ratio = observed_std / expected_std
        std_ratios.append(std_ratio)
    
    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot mean error
    ax1.plot(detailed_sample_sizes, mean_errors, 'o-', linewidth=2)
    ax1.set_title(f'Convergence of Sample Mean ({distribution_name})')
    ax1.set_xlabel('Sample Size (n)')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot std ratio
    ax2.plot(detailed_sample_sizes, std_ratios, 'o-', linewidth=2)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    ax2.set_title(f'Ratio of Observed to Expected Std of Sample Means ({distribution_name})')
    ax2.set_xlabel('Sample Size (n)')
    ax2.set_ylabel('Ratio (should approach 1.0)')
    ax2.set_xscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.show()

# Function to compare convergence rates across distributions
def compare_convergence_across_distributions():
    """Compare how quickly different distributions approach normality."""
    # Define distributions to compare
    distributions = [
        ("Uniform", np.random.uniform, (a, b), (a+b)/2, np.sqrt((b-a)**2/12)),
        ("Exponential", np.random.exponential, (1/rate,), 1/rate, 1/rate),
        ("Binomial", np.random.binomial, (n_trials, p_success), n_trials*p_success, np.sqrt(n_trials*p_success*(1-p_success))),
        ("Chi-Square", np.random.chisquare, (df,), df, np.sqrt(2*df))
    ]
    
    # Sample sizes to test
    test_sample_sizes = [2, 5, 10, 30, 50, 100]
    
    # Number of sampling repetitions
    test_repetitions = 5000
    
    # Store results - we'll use Kolmogorov-Smirnov test to measure normality
    ks_stats = {name: [] for name, _, _, _, _ in distributions}
    
    for name, generator, params, mean, std in distributions:
        # Generate population
        population = generator(*params, size=1000000)
        
        for n in test_sample_sizes:
            # Store sample means
            means = np.zeros(test_repetitions)
            
            # Generate sample means
            for i in range(test_repetitions):
                sample = np.random.choice(population, size=n)
                means[i] = np.mean(sample)
            
            # Normalize the sample means
            normalized_means = (means - np.mean(means)) / np.std(means)
            
            # Run Kolmogorov-Smirnov test against standard normal
            ks_stat, _ = stats.kstest(normalized_means, 'norm')
            ks_stats[name].append(ks_stat)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    for name in ks_stats:
        plt.plot(test_sample_sizes, ks_stats[name], 'o-', linewidth=2, label=name)
    
    plt.title('Convergence to Normality Across Different Distributions')
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Kolmogorov-Smirnov Statistic (smaller is closer to normal)')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```
![alt text](<Screenshot 2025-04-10 at 15.19.55.png>)
![alt text](<Screenshot 2025-04-10 at 15.20.20.png>)
![alt text](<Screenshot 2025-04-10 at 15.20.39.png>)
![alt text](<Screenshot 2025-04-10 at 15.21.10.png>)
![alt text](<Screenshot 2025-04-10 at 15.21.30.png>)
![alt text](<Screenshot 2025-04-10 at 15.21.42.png>)
![alt text](<Screenshot 2025-04-10 at 15.22.04.png>)
![alt text](<Screenshot 2025-04-10 at 15.22.30.png>)
![alt text](<Screenshot 2025-04-10 at 15.22.48.png>)
![alt text](<Screenshot 2025-04-10 at 15.23.05.png>)
![alt text](<Screenshot 2025-04-10 at 15.23.55.png>)
![alt text](<Screenshot 2025-04-10 at 15.24.12.png>)
![alt text](<Screenshot 2025-04-10 at 15.24.23.png>)
![alt text](<Screenshot 2025-04-10 at 15.24.35.png>)
![alt text](<Screenshot 2025-04-10 at 15.24.45.png>)
![alt text](<Screenshot 2025-04-10 at 15.24.55.png>)
![alt text](<Screenshot 2025-04-10 at 15.25.07.png>)
![alt text](<Screenshot 2025-04-10 at 15.25.21.png>)
![alt text](<Screenshot 2025-04-10 at 15.25.41.png>)
![alt text](<Screenshot 2025-04-10 at 15.25.53.png>)
![alt text](<Screenshot 2025-04-10 at 15.26.04.png>)



## Simulation Results and Analysis

### 1. Basic Sampling Distribution Analysis

For each distribution, we observed the following patterns:

#### Uniform Distribution [0, 10]
- Theoretical Mean: 5.0
- Theoretical Standard Deviation: 2.89

**Observations:**
- Even with small sample sizes (n=5), the sampling distribution already shows a bell-shaped curve
- At n=30, the sampling distribution is nearly indistinguishable from a normal distribution
- The standard error decreases proportionally to √n as expected

#### Exponential Distribution (λ=0.5)
- Theoretical Mean: 2.0
- Theoretical Standard Deviation: 2.0

**Observations:**
- The original distribution is highly right-skewed
- With n=5, the sampling distribution still shows noticeable skewness
- Only at n=30 does the sampling distribution become visibly normal
- The convergence to normality is slower than for the uniform distribution

#### Binomial Distribution (n=20, p=0.3)
- Theoretical Mean: 6.0
- Theoretical Standard Deviation: 2.05

**Observations:**
- Despite being a discrete distribution, the sampling distribution quickly approaches normality
- By n=10, the sampling distribution closely resembles a normal distribution
- The discrete nature of the original distribution has minimal impact on the convergence rate

#### Chi-Square Distribution (df=3)
- Theoretical Mean: 3.0
- Theoretical Standard Deviation: 2.45

**Observations:**
- Highly right-skewed original distribution
- Similar to the exponential distribution, convergence to normality is relatively slow
- Even at n=50, the sampling distribution maintains a slight right skew

### 2. Q-Q Plot Analysis

The Q-Q plots confirm our visual observations:

- For the uniform distribution, even at n=5, points follow the reference line closely
- For the exponential and chi-square distributions, smaller sample sizes (n=5, 10) show deviations in the tails
- As sample size increases, the Q-Q plots show progressively better alignment with the reference line for all distributions

### 3. Convergence Rate Analysis

The Kolmogorov-Smirnov (KS) test statistic provides a quantitative measure of the distance between the sampling distribution and a normal distribution. Our analysis shows:

- Symmetric distributions (uniform, binomial) converge more quickly to normality
- Skewed distributions (exponential, chi-square) require larger sample sizes
- For all distributions, the KS statistic decreases approximately proportionally to 1/√n

The mean absolute error plots demonstrate that estimation accuracy improves with the square root of the sample size, consistent with theoretical expectations.

### 4. Standard Error Verification

Our simulations confirm that the standard error of the sample mean follows the formula σ/√n:

- For all distributions, the ratio of observed to expected standard error approaches 1.0 as sample size increases
- Minor deviations at very small sample sizes (n<5) likely stem from the central limit theorem not yet fully applying

## Practical Applications

The Central Limit Theorem has significant implications across various fields:

### 1. Quality Control in Manufacturing

In manufacturing environments, products naturally exhibit variability. The CLT allows quality engineers to:
- Take small samples (typically n=5 to 30) to monitor process outputs
- Calculate control limits based on normal distribution properties
- Make statistical inferences despite not knowing the underlying distribution of individual measurements
- Detect process shifts efficiently using statistical process control charts

### 2. Financial Risk Management

Financial markets often involve non-normal distributions of returns. The CLT helps risk managers:
- Aggregate individual asset returns into portfolio returns that are more normally distributed
- Apply Value-at-Risk (VaR) models that rely on normal distribution assumptions
- Assess the reliability of mean return estimates based on sample size
- Develop more robust risk models by understanding the limitations of normality assumptions

### 3. Survey Sampling and Public Opinion Research

When conducting surveys, researchers benefit from the CLT by:
- Making inferences about population parameters from sample statistics
- Calculating confidence intervals for estimates like approval ratings or voter preferences
- Determining appropriate sample sizes to achieve desired levels of precision
- Combining multiple survey results with sound statistical methodology

### 4. Healthcare and Clinical Trials

In medical research, the CLT supports:
- Analysis of treatment effects across patient populations
- Determination of minimum sample sizes needed for statistical power
- Interpretation of biomarker measurements and laboratory test results
- Meta-analysis of multiple studies to reach more robust conclusions

## Conclusion

Our simulation study confirms that the Central Limit Theorem holds across various population distributions. The sampling distribution of the sample mean consistently approaches a normal distribution as sample size increases, with the rate of convergence depending on the characteristics of the original distribution.

Key findings include:

1. **Universal Applicability**: The CLT applies regardless of the shape of the original population distribution, though convergence rates differ.

2. **Sample Size Impact**: While the theorem technically applies as n approaches infinity, practical applications show that:
   - For symmetric distributions, n≥30 is typically sufficient for reliable normality
   - For highly skewed distributions, larger sample sizes (n≥50) may be necessary
   - The standard error decreases predictably with the square root of the sample size

3. **Distribution Characteristics**: The original distribution's shape affects convergence:
   - Symmetric distributions converge more quickly
   - Skewed distributions require larger sample sizes
   - Discrete distributions converge similarly to continuous ones

These findings have profound implications for statistical inference, allowing practitioners to make valid probabilistic statements about population parameters even when the population distribution is unknown or non-normal. This powerful property underlies countless statistical methods used across scientific disciplines and industries.

The Central Limit Theorem thus serves as a bridge between the complex, often unknown distributions of real-world phenomena and the tractable, well-understood properties of the normal distribution, enabling robust statistical inference in the face of uncertainty.