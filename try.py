import os
import glob

def delete_files(folder_path, pattern):
    files = glob.glob(os.path.join(folder_path, pattern))
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted file: {file}")
        except Exception as e:
            print(f"Error deleting file {file}: {e}")

if __name__ == "__main__":
    folder_path = "I_log_r1000/reject_diversity/4/ppo/"
    pattern = "optim"
    
    delete_files(folder_path, pattern)
    pattern = "event*"
    
    delete_files(folder_path, pattern)
