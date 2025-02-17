import os
import argparse
import json
import glob
import numpy as np
import json
import h5py
from collections import defaultdict
import pandas as pd
import concurrent.futures
import scipy.io as sio
from tifffile import imread
import tifffile
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.io as sio

def compute_euclidean_norm(data_dict):
    # print('[DEBUG]:', data_dict.items())
    # Initialize the output dictionary
    result_dict = {}
    # Iterate through each folder in the dictionary
    for folder_index, (motion_c_data, motion_r_data) in data_dict.items():
        # Compute the desired formula for each element
        result = np.sqrt((np.abs(motion_c_data) / 128)**2 + (np.abs(motion_r_data) / 50)**2) # MotionC and MotionR is combined to single array 
        # TODO: MX should the 128 and 50 be standard or dynamic according to the shape of movies? 
        # Store the result in the output dictionary
        result_dict[folder_index] = result    
    # Return the final result dictionary after processing all items
    return result_dict

def compute_percentiles(combined_motion_dict):
    # Initialize a dictionary to store unique indices
    unique_indices_dict = {}
    
    # Process each folder
    for folder_index, motion_total in combined_motion_dict.items():
        # Compute the 95th percentile
        top_5_percentile = np.percentile(motion_total, 95)
        
        # Extract indices of values above or equal to the 95th percentile
        top_5_indices = np.argwhere(motion_total >= top_5_percentile).flatten()
        
        # Compute unique indices
        unique_indices = np.unique(top_5_indices)
        
        # Store unique indices in the dictionary
        unique_indices_dict[folder_index] = unique_indices.tolist()  # Convert to list for readability
            
    return unique_indices_dict

# Common function to process .h5 and .mat files
def process_alignment_data(file_path, file_type):
    """
    Process a file (.h5 or .mat) to extract MotionC, MotionR, and metadata.

    Args:
        file_path (str): Path to the file.
        sim_description (str): Simulation description or folder name.
        file_type (str): Type of the file ('h5' or 'mat').
        model_name (str, optional): Model name for .h5 files.

    Returns:
        tuple: (sim_description, motionC, motionR, metadata, model_name)
    """
    try:
        # Initialize variables
        motion_c_data = None
        motion_r_data = None

        if file_type == 'h5':
            # Process .h5 file
            with h5py.File(file_path, 'r') as h5_file:
                # Check for MotionC at different levels
                if 'MotionC' in h5_file.keys():
                    motion_c_data = h5_file['MotionC'][:]
                elif 'C' in h5_file.keys():
                    motion_c_data = h5_file['C'][:]
                elif 'aData/motionC' in h5_file:
                    motion_c_data = h5_file['aData/motionC'][:]  # Check for aData/motionC

                # Check for MotionR at different levels
                if 'MotionR' in h5_file.keys():
                    motion_r_data = h5_file['MotionR'][:]
                elif 'R' in h5_file.keys():
                    motion_r_data = h5_file['R'][:]
                elif 'aData/motionR' in h5_file:
                    motion_r_data = h5_file['aData/motionR'][:]  # Check for aData/motionR

        elif file_type == 'mat':
            # Process .mat file
            mat_contents = sio.loadmat(file_path)

            # Assuming the structured array is stored under 'aData'
            data_key = 'aData'  # Adjust this if the key is different

            # Access the structured array
            structured_array = mat_contents[data_key]

            # Extract motionC and motionR
            motion_c_data = structured_array['motionC'][0, 0].ravel()[:]
            motion_r_data = structured_array['motionR'][0, 0].ravel()[:]

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return (motion_c_data, motion_r_data)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return (None, None if file_type == 'h5' or 'mat' else None)

# Modified functions for batched processing
def process_movie_batch(data_dir, folder_names, batch_size=5):
    """Generator that yields batches of movies from different registration methods"""
    # First collect all valid movie indices across methods
    index_map = defaultdict(list)
    
    for method in folder_names:
        method_path = os.path.join(data_dir, method)
        if os.path.exists(method_path):
            indices = [d for d in os.listdir(method_path) if d.isdigit()]
            for idx in indices:
                index_map[int(idx)].append(method)

    # Process in batches
    current_batch = []
    for idx, methods in index_map.items():
        if len(current_batch) >= batch_size:
            yield current_batch
            current_batch = []
        
        batch_entry = {'index': idx, 'methods': []}
        for method in methods:
            method_path = os.path.join(data_dir, method, str(idx))
            if os.path.exists(method_path):
                batch_entry['methods'].append(method)
        current_batch.append(batch_entry)
    
    if current_batch:
        yield current_batch

def process_single_movie(data_dir, method, idx):
    """Process a single movie from a registration method"""
    folder_path = os.path.join(data_dir, method, str(idx))
    if not os.path.exists(folder_path):
        return None

    # 1. Load alignment data
    motion_data = None
    for f in os.listdir(folder_path):
        if f.endswith(('.h5', '_ALIGNMENTDATA.mat')):
            file_type = 'h5' if f.endswith('.h5') else 'mat'
            motion_c, motion_r = process_alignment_data(os.path.join(folder_path, f), file_type)
            if motion_c is not None and motion_r is not None:
                motion_data = compute_euclidean_norm({idx: (motion_c, motion_r)})[idx]
                break

    # 2. Compute top 5% indices
    if motion_data is not None:
        top_indices = compute_percentiles({idx: motion_data})[idx]
    else:
        top_indices = []

    # 3. Load and process TIFF data in chunks
    tiff_path = None
    for f in os.listdir(folder_path):
        if f.endswith(('.tif', '.tiff')):
            tiff_path = os.path.join(folder_path, f)
            break

    if not tiff_path:
        return None

    # Memory-efficient TIFF reading
    with tifffile.TiffFile(tiff_path) as tif:
        num_frames = len(tif.pages)
        mean_frame = np.zeros(tif.pages[0].shape, dtype=np.float32)
        
        # Compute mean frame
        for i, page in enumerate(tif.pages):
            mean_frame += page.asarray().astype(np.float32) / num_frames #TODO: MX is this okay? Better than taking NaN mean?

        # Get high-motion frames
        high_motion_frames = []
        for i in top_indices:
            if i < num_frames:
                high_motion_frames.append(tif.pages[i].asarray())

    return {
        'method': method,
        'index': idx,
        'mean_frame': mean_frame,
        'high_motion_frames': high_motion_frames,
        'top_indices': top_indices
    }

def copy_folder_structure(src, dst, folder_names):
    # Validate which folders contain at least one file of the specified type
    valid_folders = set()
    for folder, ext in folder_names.items():
        folder_path = os.path.join(src, folder)
        if not os.path.isdir(folder_path):
            continue
        ext = f".{ext}" if not ext.startswith('.') else ext
        if any(f.endswith(ext) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))):
            valid_folders.add(folder)
    
    # Copy directory structure for valid folders and their subdirectories
    for root, dirs, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        if any(rel_path.startswith(folder) for folder in valid_folders):
            dest_dir = os.path.join(dst, rel_path)
            os.makedirs(dest_dir, exist_ok=True)

def run(data_dir, output_path):
    folder_names = {
        'suite2p': 'h5',
        'caiman_stripCaiman': 'h5',
        'stripRegisteration_matlab': 'mat',
        'stripRegisteration': 'h5'
    }

    # Create output directory structure
    copy_folder_structure(data_dir, output_path, folder_names)

    # Dictionary to store correlations for each method
    method_correlations = defaultdict(list)

    # Process in batches
    batch_gen = process_movie_batch(data_dir, folder_names)
    
    for batch in batch_gen:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for movie in batch:
                for method in movie['methods']:
                    futures.append(executor.submit(
                        process_single_movie,
                        data_dir, 
                        method,
                        movie['index']
                    ))

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if not result:
                    continue
                
                # Compute correlations for this movie
                corr_values = []
                for frame in result['high_motion_frames']:
                    corr, _ = pearsonr(
                        np.nan_to_num(result['mean_frame'].flatten(), nan=0.0),
                        np.nan_to_num(frame.flatten(), nan=0.0)
                    )
                    corr_values.append(corr)

                # Save results immediately to free memory
                save_dir = os.path.join(output_path, result['method'], str(result['index']))
                os.makedirs(save_dir, exist_ok=True)

                # Save correlation data
                with open(os.path.join(save_dir, 'correlations.json'), 'w') as f:
                    json.dump(corr_values, f)

                # Save visualization with mean correlation in title
                plt.figure(figsize=(10, 6))
                plt.hist(corr_values, bins=50, edgecolor='black')
                mean_corr = np.mean(corr_values)
                plt.title(f'Correlation Histogram\nMethod: {result["method"]} Index: {result["index"]}\nMean Correlation: {mean_corr:.4f}')
                plt.xlabel('Pearson Correlation Coefficient')
                plt.ylabel('Frequency')
                plt.savefig(os.path.join(save_dir, 'correlation_histogram.png'))
                plt.close()

                # Store correlations for summary plot
                method_correlations[result['method']].extend(corr_values)

    # Generate summary plots for each method
    for method, correlations in method_correlations.items():
        summary_dir = os.path.join(output_path, method)
        os.makedirs(summary_dir, exist_ok=True)

        # Save summary histogram
        plt.figure(figsize=(10, 6))
        plt.hist(correlations, bins=50, edgecolor='black')
        overall_mean_corr = np.mean(correlations)
        plt.title(f'Summary Correlation Histogram\nMethod: {method}\nMean Correlation Across Movies: {overall_mean_corr:.4f}')
        plt.xlabel('Pearson Correlation Coefficient')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(summary_dir, 'summary_correlation_histogram.png'))
        plt.close()

        # Save mean correlation value to a text file
        with open(os.path.join(summary_dir, 'mean_correlation.txt'), 'w') as f:
            f.write(f'Mean Correlation Across Movies: {overall_mean_corr:.4f}\n')


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input to the folder that contains tiff files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="../results/",
        help="Output folder to save the results.",
    )

    # Parse the arguments
    args = parser.parse_args()
    # Assign the parsed arguments to params dictionary

    run(args.input, args.output)




