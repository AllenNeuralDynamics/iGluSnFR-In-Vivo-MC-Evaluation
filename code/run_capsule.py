import os
import argparse
import json
import numpy as np
import h5py
import pandas as pd
import concurrent.futures
import scipy.io as sio
from tifffile import imread
import matplotlib.pyplot as plt
import scipy.io as sio

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


def read_tiff_h5_aData(data_dir, file_type):
    # Initialize an empty dictionary to store TIFF data
    tiff_dict = {}
    aData_dict = {}
    # Iterate through numerical subfolders
    for folder in sorted(os.listdir(data_dir)):
        if folder.isdigit():  # Ensure the folder name is numeric
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):  # Check if it is a directory
                # Find TIFF files in the folder
                for file in os.listdir(folder_path):
                    if file.endswith('.tif') or file.endswith('.tiff'):  # Check for TIFF file extensions
                        tiff_path = os.path.join(folder_path, file)
                        # Read the TIFF file
                        tiff_data = imread(tiff_path)
                        # Append to dictionary with key as folder number
                        tiff_dict[int(folder)] = tiff_data
                    if file.endswith('.h5'):
                        h5_path = os.path.join(folder_path, file)
                        aData_dict[int(folder)] = process_alignment_data(h5_path, file_type) # Get aData from .h5 file
                    if file.endswith('_ALIGNMENTDATA.mat'):
                        mat_path = os.path.join(folder_path, file)
                        aData_dict[int(folder)] = process_alignment_data(mat_path, file_type) # Get aData from mat file
    return tiff_dict, aData_dict

def compute_euclidean_norm(data_dict):
    # print('[DEBUG]:', data_dict.items())
    # Initialize the output dictionary
    result_dict = {}
    # Iterate through each folder in the dictionary
    for folder_index, (motion_c_data, motion_r_data) in data_dict.items():
        # Compute the desired formula for each element
        result = np.sqrt((np.abs(motion_c_data) / 128)**2 + (np.abs(motion_r_data) / 50)**2) # MotionC and MotionR is combined to single array        
        # Store the result in the output dictionary
        result_dict[folder_index] = result    
    # Return the final result dictionary after processing all items
    return result_dict

def compute_percentiles(combined_motion_dict):
    # Initialize dictionaries to store results
    top_5_values_dict = {}
    top_5_indices_dict = {}
    
    # Process each folder
    for folder_index, motion_total in combined_motion_dict.items():
        # Compute the 95th percentile
        top_5_percentile = np.percentile(motion_total, 95)
        
        # Extract values above or equal to the 95th percentile
        top_5_values = motion_total[motion_total >= top_5_percentile]
        
        # Extract indices of values above or equal to the 95th percentile
        top_5_indices = np.argwhere(motion_total >= top_5_percentile)
        
        # Store results in dictionaries
        top_5_values_dict[folder_index] = top_5_values
        top_5_indices_dict[folder_index] = top_5_indices
            
    return top_5_indices_dict

def run(data_dir, output_path):
    # Create output directory
    if not os.path.exists(output_path):
        print("Creating output directory...")
        os.makedirs(output_path)
    print("Output directory created at", output_path)

    if 'suite2p' in data_dir:
        tiff_dict_suite2p, aData_dict_suite2p = read_tiff_h5_aData(data_dir, 'h5')
        motion_total_suite2p_dict = compute_euclidean_norm(aData_dict_suite2p)
        top_5_indices_dict_suite2p = compute_percentiles(motion_total_suite2p_dict)
        print('Done suite2p')

    if 'caiman_stripCaiman' in data_dir:
        tiff_dict_caiman, aData_dict_caiman = read_tiff_h5_aData(data_dir, 'h5')
        motion_total_caiman_dict = compute_euclidean_norm(aData_dict_caiman)
        top_5_indices_dict_caiman = compute_percentiles(motion_total_caiman_dict)
        print('Done caiman')

    if 'stripRegisteration_matlab' in data_dir:
        tiff_dict_strip_matlab, aData_dict_strip_matlab = read_tiff_h5_aData(data_dir, 'mat')
        # motion_total_strip_matlab_dict = compute_euclidean_norm(aData_dict_strip_matlab)
        # top_5_indices_dict_strip_matlab = compute_percentiles(motion_total_strip_matlab_dict)
        # print(top_5_indices_dict_strip_matlab)
        # print('Done matlab')

    if 'stripRegisteration' in data_dir:
        tiff_dict_strip, aData_dict_strip = read_tiff_h5_aData(data_dir, 'h5')
        motion_total_strip_dict = compute_euclidean_norm(aData_dict_strip)
        top_5_indices_dict_strip = compute_percentiles(motion_total_strip_dict)
        print('Done StripRegisteration')




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