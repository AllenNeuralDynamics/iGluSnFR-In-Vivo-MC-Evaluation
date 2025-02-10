import os
import argparse
import json
import glob
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
    """
    Reads TIFF files and alignment data from a directory structure.

    Args:
        data_dir (str): Path to the main directory containing subfolders.
        file_type (str): File type for alignment data processing ('h5' or 'mat').

    Returns:
        tuple: A tuple containing two dictionaries:
            - tiff_dict (dict): Dictionary of TIFF data, with folder number as keys.
            - aData_dict (dict): Dictionary of alignment data, with folder number as keys.
    """

    tiff_dict = {}
    aData_dict = {}

    for folder in sorted(os.listdir(data_dir)):
        if not folder.isdigit():  # Skip non-numeric folders immediately
            continue

        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):  # Ensure it's a directory
            continue

        folder_num = int(folder)  # Convert folder name to integer for dictionary keys

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)  # Full path to the file

            if file.endswith(('.tif', '.tiff')):  # Check for TIFF files
                if "DOWNSAMPLED" in file or "suite2p" in folder_path.lower() or "caiman" in folder_path.lower():
                    try:
                        tiff_data = imread(file_path)
                        # print('tiff_data:', folder, tiff_data.shape)
                        tiff_dict[folder_num] = tiff_data
                    except Exception as e:
                        print(f"Error reading TIFF file {file_path}: {e}")

            elif file.endswith('.h5'):
                aData_dict[folder_num] = process_alignment_data(file_path, file_type)
            elif file.endswith('_ALIGNMENTDATA.mat'):
                aData_dict[folder_num] = process_alignment_data(file_path, file_type)

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

def process_folder(data_dir, folder_name, file_extension):
    folder_path = os.path.join(data_dir, folder_name)
    if os.path.isdir(folder_path):
        tiff_dict, aData_dict = read_tiff_h5_aData(folder_path, file_extension) # 1. Extract tiff and alignment data (motionC and motionR)
        motion_total_dict = compute_euclidean_norm(aData_dict) #2. Combine motionC and motionR by  normalized Euclidean norm by computing the magnitude of a vector with the two components
        top_5_indices_dict = compute_percentiles(motion_total_dict) #3. Get the top 5% with the highest motion
        
        print(f'Done {folder_name}')
        return top_5_indices_dict, tiff_dict

def run(data_dir, output_path):
    # Create output directory
    if not os.path.exists(output_path):
        print("Creating output directory...")
        os.makedirs(output_path)
    print("Output directory created at", output_path)

    folder_names = {
    'suite2p': 'h5',
    'caiman_stripCaiman': 'h5',
    'stripRegisteration_matlab': 'mat',
    'stripRegisteration': 'h5'
    }

    # Use glob to find directories matching the names
    found_folders = [os.path.basename(x) for x in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(x)]

    methods_indices = {}
    tiff_indices = {}
    
    for folder, ext in folder_names.items():
        if folder in found_folders:
            folder_path = os.path.join(data_dir, folder)
            methods_indices[folder], tiff_indices[folder]  = process_folder(data_dir, folder, ext)

    # Flatten nested indices
    all_indices = []
    for indices_dict in methods_indices.values():
        if isinstance(indices_dict, dict):
            # Flatten individual method indices
            for indices in indices_dict.values():
                all_indices.extend(indices)  # Add individual elements
        else:
            print(f"Warning: Expected dict but got {type(indices_dict)}")

    # Now works with homogeneous array
    unique_indices, counts = np.unique(all_indices, return_counts=True)

    # tiff_indices contains 1st methods used for registration. Each methods contains the registered movie. 
    # If 3 methods are used (Suite2p, caiman, Strip). Suite2p contains eg 9 registered movies. 
    # print('tiff_indices', tiff_indices.shape)
    mean_registered_movies = {}
    frames_with_motion = {}
    for folder, indices in tiff_indices.items():
        mean_registered_movies[folder] = {}
        frames_with_motion[folder] = {}
        for index, tiff_array in indices.items():
            # print(folder,index, tiff_array.shape)
            mean_registered_movies[folder][index] = np.mean(tiff_array, axis=0)
            frames_with_motion[folder][index] = tiff_array[unique_indices[index], :, :]
    print(frames_with_motion)

            

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




