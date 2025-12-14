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

    
def save_data(cleaned_df, src_path, output_path):
    """
    Save a pandas DataFrame to a CSV file.

    Parameters:
    data (pd.DataFrame): The DataFrame to save.
    file_path (str): The path where the CSV file will be saved.
    """
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    for person in cleaned_df['name'].unique():
        src_path_person = os.path.join(src_path, person)
        out_path_person = os.path.join(output_path, "/"+person)

        if os.path.isdir(src_path_person) is False:
            print(f"[WARNING] No folder found for: {person}")
            continue
        if os.path.exists(out_path_person):
            print(f"[WARNING] Folder already exists for: {person}")
            continue
        
        shutil.copytree(src_path_person, out_path_person)
