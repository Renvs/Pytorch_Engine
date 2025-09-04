import os
import subprocess
import sys

from pathlib import Path

def setup_git_repository(repo_url: str, repo_dir: str) -> None:
    """
    Clone or update a git repository.
    
    Args:
        repo_url: URL of the git repository
        repo_dir: Local directory name for the repository
    """
    if not os.path.isdir(repo_dir):
        print(f"[INFO] Cloning repository: {repo_url}")
        subprocess.run(["git", "clone", repo_url], check=True)
    else:
        print(f"[INFO] Repository '{repo_dir}' found. Checking for updates...")
        subprocess.run(["git", "-C", repo_dir, "fetch"], check=True)

        status_result = subprocess.run(
            ["git", "-C", repo_dir, "status", "-uno"],
            capture_output=True, text=True, check=True
        )

        if "Your branch is behind" in status_result.stdout:
            print("[INFO] New changes detected. Pulling from remote...")
            subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
        else:
            print("[INFO] Repository is up to date.")

def add_module_path(module_path: str) -> None:
    """
    Add module path to system path if not already present.
    
    Args:
        module_path: Path to the module directory
    """
    if module_path not in sys.path:
        sys.path.append(module_path)
        print(f"[INFO] Added {module_path} to system path")

# def main():
#     """Main function to set up the repository and import modules."""
#     # Configuration
#     repo_url = "https://github.com/Renvs/Pytorch_Engine"
#     repo_dir = "Pytorch_Engine"
    
#     # Setup repository
#     setup_git_repository(repo_url, repo_dir)
    
#     # Import modules
#     success = import_modules(repo_dir)
    
#     if success:
#         print("[INFO] Setup completed successfully!")
#         # You can now use the imported modules
#         # Example: engine.train_model(...), data_setup.create_dataloaders(...), etc.
#     else:
#         print("[ERROR] Setup failed. Please check the repository structure.")

# if __name__ == "__main__":
#     main()