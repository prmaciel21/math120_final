import os 
import pandas as pd
import shutil

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """
    Clean the input DataFrame by keeping only rows where the image count
    is between 20 and 30 (inclusive).

    Parameters:
    data (pd.DataFrame): Must contain a column named 'image'.

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    #Remove missing values first (optional)
    data = data.dropna()

    #Filter based on image count
    cleaned_df = data[(data['images'] >= 20) & (data['images'] <= 30)]

    return cleaned_df

def check_if_dir_exists(parent, dir_name):
    """
    Check if a directory exists.

    Parameters:
    directory (str): The path to the directory.

    Returns:
    bool: True if the directory exists, False otherwise.
    """
    target_path = os.path.join(parent, dir_name)

    # Check if the path exists AND is a directory
    if os.path.isdir(target_path):
        print(f"'{target_path}' exists and is a directory.")
        return True
    else:
        print(f"'{target_path}' does not exist or is not a directory.")
        return False
    
def save_data(cleaned_df, src_path, output_path):
    """
    Save a pandas DataFrame to a CSV file.

    Parameters:
    data (pd.DataFrame): The DataFrame to save.
    file_path (str): The path where the CSV file will be saved.
    """
    if os.path.isdir(src_path) is False:
        raise FileNotFoundError(f"The source path at {src_path} does not exist.")
    for person in cleaned_df['name']:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        if check_if_dir_exists(src_path, person) is False:
            print(f"[WARNING] No folder found for: {person}")
            continue
        else:
            shutil.copytree(os.path.join(src_path, person), os.path.join(output_path+'/', person))
