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

    
def save_data(cleaned_df, src_path, output_path, min_imgs=20, max_imgs=30):
    """
    Copy person image folders from src_path to output_path
    if they contain between min_imgs and max_imgs images.
    """

    os.makedirs(output_path, exist_ok=True)
    copied = 0

    for name in cleaned_df['name'].unique():
        person = name.replace(" ", "_")  # 🔑 FIX

        src_path_person = os.path.join(src_path, person)
        out_path_person = os.path.join(output_path, person)

        if not os.path.isdir(src_path_person):
            print(f"[SKIP] No folder found: {person}")
            continue

        img_count = len([
            f for f in os.listdir(src_path_person)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if not (min_imgs <= img_count <= max_imgs):
            print(f"[SKIP] {person}: {img_count} images")
            continue

        if os.path.exists(out_path_person):
            print(f"[SKIP] Exists: {person}")
            continue

        shutil.copytree(src_path_person, out_path_person)
        copied += 1
        print(f"[COPIED] {person}: {img_count} images")
