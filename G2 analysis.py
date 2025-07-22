import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import trapz
import glob
import os

def parse_data_file(filename):
    """
    Parse HBT measurement data file with mixed units (p, n suffixes)
    """
    data = []
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num == 0:  # Skip header
                continue
                
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    # Parse time with unit suffixes
                    time_str = parts[0].replace(',', '.')  # Handle European decimal format
                    
                    # Convert time units to nanoseconds
                    if time_str.endswith('p'):
                        time_val = float(time_str[:-1]) / 1000  # picoseconds to nanoseconds
                    elif time_str.endswith('n'):
                        time_val = float(time_str[:-1])  # already in nanoseconds
                    else:
                        time_val = float(time_str)  # assume nanoseconds
                    
                    # Parse counts (handle European decimal format)
                    counts_str = parts[1].replace(',', '.')
                    counts_val = float(counts_str)
                    
                    data.append([time_val, counts_val])
                    
                except ValueError:
                    print(f"Warning: Could not parse line {line_num + 1} in {filename}: {line.strip()}")
                    continue
    
    return np.array(data)

def gaussian_plus_constant(x, A, mu, sigma, C):
    """
    Gaussian peak plus constant background
    A: amplitude of Gaussian
    mu: center of Gaussian
    sigma: width of Gaussian
    C: constant background level
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + C

def analyze_hbt_data(time, counts, plot=True, title="HBT Measurement"):
    """
    Analyze HBT data to separate SPDC peak from accidental background
    """
    
    # Initial parameter guesses
    background_estimate = np.mean([counts[:5], counts[-5:]])  # Average of edges
    peak_estimate = np.max(counts) - background_estimate
    center_estimate = time[np.argmax(counts)]
    width_estimate = 1.0  # Initial guess for width in nanoseconds
    
    initial_guess = [peak_estimate, center_estimate, width_estimate, background_estimate]
    
    try:
        # Fit Gaussian + constant
        popt, pcov = curve_fit(gaussian_plus_constant, time, counts, 
                             p0=initial_guess, maxfev=5000)
        
        A, mu, sigma, C = popt
        
        # Calculate fitted curves
        fitted_total = gaussian_plus_constant(time, A, mu, sigma, C)
        gaussian_only = gaussian_plus_constant(time, A, mu, sigma, 0)
        background_only = np.full_like(time, C)
        
        # Calculate coincidences from histogram data
        # Background coincidences (accidental) - use fitted background level
        time_range = time[-1] - time[0]
        accidental_coincidences = C * time_range
        
        # SPDC coincidences: sum experimental data in Gaussian region minus background
        # Define Gaussian region as center ± 3*sigma (contains ~99.7% of Gaussian)
        gaussian_region_mask = np.abs(time - mu) <= 3 * abs(sigma)
        
        # Sum experimental counts in Gaussian region
        counts_in_region = np.sum(counts[gaussian_region_mask])
        
        # Calculate background contribution in the same region
        time_steps_in_region = np.sum(gaussian_region_mask)
        if len(time) > 1:
            dt = (time[-1] - time[0]) / (len(time) - 1)  # time step
            background_in_region = C * time_steps_in_region * dt
        else:
            background_in_region = 0
        
        # SPDC coincidences = total counts in region - background in region
        spdc_coincidences = counts_in_region - background_in_region
        
        total_coincidences = spdc_coincidences + accidental_coincidences
        
        if plot:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(time, counts, 'bo-', label='Data', markersize=4)
            plt.plot(time, fitted_total, 'r-', linewidth=2, label='Total Fit')
            plt.plot(time, gaussian_only, 'g--', linewidth=2, label='SPDC (Gaussian)')
            plt.plot(time, background_only, 'orange', linestyle=':', linewidth=2, label='Accidental Background')
            
            # Highlight the Gaussian region used for SPDC calculation
            gaussian_region_mask = np.abs(time - mu) <= 3 * abs(sigma)
            plt.fill_between(time[gaussian_region_mask], 0, counts[gaussian_region_mask], 
                           alpha=0.3, color='lightgreen', label='SPDC Region (±3σ)')
            
            plt.xlabel('Time (ns)')
            plt.ylabel('Events per second')
            plt.title(f'{title} - HBT Correlation')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Residuals plot
            plt.subplot(2, 1, 2)
            residuals = counts - fitted_total
            plt.plot(time, residuals, 'ro-', markersize=3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            plt.xlabel('Time (ns)')
            plt.ylabel('Residuals')
            plt.title('Fit Residuals')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Print results
        print(f"\n{title} Analysis Results:")
        print(f"Gaussian center (μ): {mu:.3f} ns")
        print(f"Gaussian width (σ): {abs(sigma):.3f} ns")
        print(f"Gaussian amplitude (A): {A:.1f}")
        print(f"Background level (C): {C:.1f} events/s")
        print(f"Gaussian region: {mu - 3*abs(sigma):.3f} to {mu + 3*abs(sigma):.3f} ns")
        print(f"Data points in Gaussian region: {time_steps_in_region}")
        print(f"Total counts in Gaussian region: {counts_in_region:.0f}")
        print(f"Background in Gaussian region: {background_in_region:.0f}")
        print(f"SPDC coincidences (experimental): {spdc_coincidences:.0f}")
        print(f"Total accidental coincidences: {accidental_coincidences:.0f}")
        print(f"Total coincidences: {total_coincidences:.0f}")
        print(f"Signal-to-background ratio: {spdc_coincidences/accidental_coincidences:.2f}")
        
        return {
            'spdc_coincidences': spdc_coincidences,
            'accidental_coincidences': accidental_coincidences,
            'total_coincidences': total_coincidences,
            'fit_params': popt,
            'fit_errors': np.sqrt(np.diag(pcov)),
            'gaussian_center': mu,
            'gaussian_width': abs(sigma),
            'background_level': C,
            'signal_to_background': spdc_coincidences/accidental_coincidences
        }
        
    except Exception as e:
        print(f"Error fitting data for {title}: {e}")
        return None

def process_multiple_files(file_pattern="*.txt"):
    """
    Process multiple HBT data files and summarize results
    """
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    print(f"Found {len(files)} files to process")
    
    results_summary = []
    
    for i, filename in enumerate(files):
        print(f"\n{'='*60}")
        print(f"Processing file {i+1}/{len(files)}: {filename}")
        print('='*60)
        
        try:
            data = parse_data_file(filename)
            
            if len(data) == 0:
                print(f"No valid data found in {filename}")
                continue
                
            time = data[:, 0]
            counts = data[:, 1]
            
            # Analyze this file
            result = analyze_hbt_data(time, counts, plot=True, 
                                    title=f"{os.path.basename(filename)}")
            
            if result:
                result['filename'] = filename
                results_summary.append(result)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Summary of all files
    if results_summary:
        print(f"\n{'='*80}")
        print("SUMMARY OF ALL FILES")
        print('='*80)
        
        df_summary = pd.DataFrame(results_summary)
        
        print(f"\n{'File':<30} {'SPDC':<12} {'Accidental':<12} {'Total':<12} {'S/B Ratio':<10}")
        print("-" * 80)
        
        total_spdc = 0
        total_accidental = 0
        
        for result in results_summary:
            filename = os.path.basename(result['filename'])
            spdc = result['spdc_coincidences']
            accidental = result['accidental_coincidences']
            total = result['total_coincidences']
            ratio = result['signal_to_background']
            
            print(f"{filename:<30} {spdc:<12.0f} {accidental:<12.0f} {total:<12.0f} {ratio:<10.2f}")
            
            total_spdc += spdc
            total_accidental += accidental
        
        print("-" * 80)
        print(f"{'TOTALS':<30} {total_spdc:<12.0f} {total_accidental:<12.0f} {total_spdc + total_accidental:<12.0f} {total_spdc/total_accidental:<10.2f}")
        
        # Statistics
        spdc_values = [r['spdc_coincidences'] for r in results_summary]
        print(f"\nStatistics for SPDC coincidences:")
        print(f"Mean: {np.mean(spdc_values):.0f} ± {np.std(spdc_values):.0f}")
        print(f"Range: {np.min(spdc_values):.0f} - {np.max(spdc_values):.0f}")
        
    return results_summary

# Example usage:
if __name__ == "__main__":
    # Process all .txt files in current directory
    #results = process_multiple_files("*.txt")
    
    # Or process a specific file:
    data = parse_data_file("asdasd.txt")
    time = data[:, 0]
    counts = data[:, 1]
    result = analyze_hbt_data(time, counts, plot=True, title="Sample HBT Data")