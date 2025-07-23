# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 10:06:01 2025

@author: jr.mejia1228
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import trapz
import glob
import os
import re

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

def extract_temperature_from_filename(filename):
    """
    Extract temperature from filename like G2-52s-33C.txt -> 33
    """
    # Extract just the filename without path
    basename = os.path.basename(filename)
    
    # Look for pattern like "33C" or "33c"
    match = re.search(r'(\d+)[Cc]', basename)
    if match:
        return float(match.group(1))
    
    # If no temperature found, return None
    print(f"Warning: Could not extract temperature from filename: {basename}")
    return None

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
        # Define Gaussian region as center ± 3*sigma (contains ~99.7% of Gaussian)
        gaussian_region_mask = np.abs(time - mu) <= 3 * abs(sigma)
        
        # SPDC coincidences: sum experimental data in Gaussian region minus background
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
        
        # Accidental coincidences: sum all experimental data minus SPDC
        total_counts = np.sum(counts)
        accidental_coincidences = total_counts - spdc_coincidences
        
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
        print(f"Signal-to-background ratio: {spdc_coincidences/background_in_region:.2f}")
        
        return {
            'spdc_coincidences': spdc_coincidences,
            'background_in_region': background_in_region,
            'counts_in_region': counts_in_region,
            'fit_params': popt,
            'fit_errors': np.sqrt(np.diag(pcov)),
            'gaussian_center': mu,
            'gaussian_width': abs(sigma),
            'background_level': C,
            'signal_to_background': spdc_coincidences/background_in_region
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
            
            # Extract temperature from filename
            temperature = extract_temperature_from_filename(filename)
            
            # Analyze this file
            result = analyze_hbt_data(time, counts, plot=True, 
                                    title=f"{os.path.basename(filename)}")
            
            if result:
                result['filename'] = filename
                result['temperature'] = temperature
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
        
        print(f"\n{'File':<30} {'Temp(°C)':<10} {'SPDC':<12} {'Accidental':<12} {'Total':<12} {'S/B Ratio':<10}")
        print("-" * 90)
        
        total_spdc = 0
        total_accidental = 0
        
        for result in results_summary:
            filename = os.path.basename(result['filename'])
            temp = result['temperature'] if result['temperature'] is not None else 'N/A'
            spdc = result['spdc_coincidences']
            accidental = result['background_in_region']
            total = result['counts_in_region']
            ratio = result['signal_to_background']
            
            print(f"{filename:<30} {temp!s:<10} {spdc:<12.0f} {accidental:<12.0f} {total:<12.0f} {ratio:<10.2f}")
            
            total_spdc += spdc
            total_accidental += accidental
        
        print("-" * 90)
        print(f"{'TOTALS':<30} {'':<10} {total_spdc:<12.0f} {total_accidental:<12.0f} {total_spdc + total_accidental:<12.0f} {total_spdc/total_accidental:<10.2f}")
        
        # Plot temperature dependence
        plot_temperature_dependence(results_summary)
        
    return results_summary

def plot_temperature_dependence(results_summary):
    """
    Plot SPDC, accidental, and total coincidences as a function of temperature
    """
    # Filter out results without temperature data
    temp_results = [r for r in results_summary if r['temperature'] is not None]
    
    if len(temp_results) == 0:
        print("No temperature data found in filenames for plotting")
        return
    
    # Extract data for plotting
    temperatures = [r['temperature'] for r in temp_results]
    spdc_counts = [r['spdc_coincidences'] for r in temp_results]
    accidental_counts = [r['background_in_region'] for r in temp_results]
    total_counts = [r['counts_in_region'] for r in temp_results]
    
    # Sort by temperature
    sorted_data = sorted(zip(temperatures, spdc_counts, accidental_counts, total_counts))
    temperatures, spdc_counts, accidental_counts, total_counts = zip(*sorted_data)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot 1: All three quantities
    plt.subplot(2, 2, 1)
    plt.plot(temperatures, spdc_counts, 'go-', linewidth=2, markersize=8, label='SPDC Coincidences')
    plt.plot(temperatures, accidental_counts, 'ro-', linewidth=2, markersize=8, label='Accidental Coincidences')
    plt.plot(temperatures, total_counts, 'bo-', linewidth=2, markersize=8, label='Total Coincidences')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Coincidences')
    plt.title('Coincidences vs Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: SPDC only
    plt.subplot(2, 2, 2)
    plt.plot(temperatures, spdc_counts, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('SPDC Coincidences')
    plt.title('SPDC Coincidences vs Temperature')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Accidental only
    plt.subplot(2, 2, 3)
    plt.plot(temperatures, accidental_counts, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Accidental Coincidences')
    plt.title('Accidental Coincidences vs Temperature')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Signal-to-background ratio
    plt.subplot(2, 2, 4)
    ratios = [s/a for s, a in zip(spdc_counts, accidental_counts)]
    plt.plot(temperatures, ratios, 'mo-', linewidth=2, markersize=8)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Signal-to-Background Ratio')
    plt.title('S/B Ratio vs Temperature')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print temperature analysis
    print(f"\n{'='*60}")
    print("TEMPERATURE ANALYSIS")
    print('='*60)
    print(f"Temperature range: {min(temperatures):.1f}°C to {max(temperatures):.1f}°C")
    print(f"Number of temperature points: {len(temperatures)}")
    
    # Find temperature with maximum SPDC
    max_spdc_idx = spdc_counts.index(max(spdc_counts))
    print(f"Maximum SPDC at {temperatures[max_spdc_idx]:.1f}°C: {spdc_counts[max_spdc_idx]:.0f} coincidences")
    
    # Find temperature with maximum S/B ratio
    max_ratio_idx = ratios.index(max(ratios))
    print(f"Maximum S/B ratio at {temperatures[max_ratio_idx]:.1f}°C: {ratios[max_ratio_idx]:.2f}")

# Example usage:
if __name__ == "__main__":
    # Process all .txt files in current directory
    results = process_multiple_files("Experimental Results/2025-07-22/Medicion G2 vs T\*.txt")
    
    # Or process a specific file:
    # data = parse_data_file("G2-52s-33C.txt")
    # time = data[:, 0]
    # counts = data[:, 1]
    # result = analyze_hbt_data(time, counts, plot=True, title="Sample HBT Data")