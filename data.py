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
    src_path = os.path.abspath(src_path)
    output_path = os.path.abspath(output_path)

    os.makedirs(output_path, exist_ok=True)

    for name in cleaned_df['name'].unique():
        person = name.replace(" ", "_")

        src_person = os.path.join(src_path, person)
        out_person = os.path.join(output_path, person)  # NO leading slash

        if not os.path.isdir(src_person):
            continue

        if os.path.isdir(out_person):
            print(f"[SKIP] Exists in output: {person}")
            continue

        img_count = len([
            f for f in os.listdir(src_person)
            if f.lower().endswith(".jpg")
        ])

        if not (min_imgs <= img_count <= max_imgs):
            continue

        shutil.copytree(src_person, out_person)
        print(f"[COPIED] {person}")

