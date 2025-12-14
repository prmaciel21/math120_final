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
    for person in cleaned_df['name'].unique():
        person_src_folder = os.path.join(src_path, person)
        person_out_folder = os.path.join(output_path, person)
        
        if not os.path.exists(person_src_folder):
            print(f"[WARNING] No folder found for: {person}")
            continue

        os.makedirs(person_out_folder, exist_ok=True)

        img_files = os.listdir(person_src_folder)
        img_files = [f for f in img_files if f.lower().endswith(".jpg")]

        for img in img_files:
            src = os.path.join(person_src_folder, img)
            dst = os.path.join(person_out_folder, img)
            shutil.copy2(src, dst)

        print(f"[COPIED] {len(img_files)} images for: {person}")