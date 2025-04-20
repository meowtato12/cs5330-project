import os
import shutil

# Helper function to clear all contents inside a directory
def clear_directory_contents(dir_path):
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Delete file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete subdirectory
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

# Main function to clean specific folders used for generated data
def clean_generated_folders():
    folders_to_clear = [
        "data/data_processing/train/image",
        "data/data_processing/train/labels",
        "data/data_processing/test/image",
        "data/data_processing/test/labels",
        "data/data_processing/visualized_images",  # For annotation visualizations
        "forecast_images"  # For prediction visualizations
    ]

    for folder in folders_to_clear:
        if os.path.exists(folder):
            print(f"Clearing contents of: {folder}")
            clear_directory_contents(folder)
        else:
            print(f"Folder does not exist, skipping: {folder}")

# Execute cleanup when run as a script
if __name__ == "__main__":
    clean_generated_folders()

