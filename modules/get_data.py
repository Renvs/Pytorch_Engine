import requests
import zipfile
import os

from pathlib import Path

def get_data(url: str, data_path: str, save_path: str, filename: str = None):
    try:
        if save_path is not None:
            print(f'[INFO] Checking Directory at {save_path}')
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            model_path = save_path / 'best_model.pt'

        if data_path is not None and filename is not None:
            print(f'[INFO] Checking Directory at {data_path}')
            data_path = Path(data_path)
            data_path.mkdir(parents=True, exist_ok=True)
            file_path = data_path / filename

            if url is not None:
                with open(file_path.with_suffix('.zip'), 'wb') as f:        
                    print(f'[INFO] Downloading fRom {url} at {file_path}')
                    request = requests.get(url, stream=True)
                    f.write(request.content)
            else: 
                raise ValueError('[ERROR] No URL provided')

            with zipfile.ZipFile(file_path.with_suffix('.zip'), 'r') as zip_file:
                print(f'[INFO] Extracting from {file_path}')
                zip_file.extractall(file_path)
            
            if os.path.exists(file_path.with_suffix('.zip')):
                os.remove(file_path.with_suffix('.zip'))

        else:
            raise ValueError('[ERROR] No data path / filename provided')

    except ValueError as e:
        print(e)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] A network error occurred: {e}")
    except zipfile.BadZipFile:
        print(f"[ERROR] Failed to unzip file. It may be corrupted or not a valid zip file.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

    train_dir = file_path / 'train'
    test_dir = file_path / 'test'

    return train_dir, test_dir, model_path

# =============== Legacy ===============

# def get_data(url: str, data_path: Path = DATA_PATH):
#     """
#     Downloads and unzips data from a URL into a specified path.

#     The pipeline is as follows:
#     1. Derives the filename from the URL.
#     2. Checks if the URL points to a .zip file.
#     3. Creates a destination directory named after the zip file (without extension).
#     4. Checks if the directory is already populated to avoid re-downloading.
#     5. Downloads the zip file.
#     6. Extracts the contents.
#     7. Removes the temporary zip file.
#     """
#     try:
#         zip_filename = os.path.basename(urllib.parse.urlparse(url).path)
#         file_path = data_path / zip_filename

#         if zip_filename.lower().endswith('.zip'):
#             extract_name = zip_filename[:-4]
#         else:
#             extract_name = zip_filename

#         extract_path = data_path / extract_name

#         if extract_path.is_dir():
#             raise ValueError(f'[ERROR]  Directory already exists at {extract_path}')
        
#         print(f'[INFO] Ensuring directory at {extract_path} is exist')
#         extract_path.mkdir(parents=True, exist_ok=True)

#         if any(extract_path.iterdir()):
#             print(f'[INFO]  Directory is not empty at {extract_path}, skipping download')
#             return
        
#         print(f'[INFO] Downloading from {url} at {file_path}')
#         with requests.get(url, stream=True) as r:
#             r.raise_for_status()
#             with open(file_path, 'wb') as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     f.write(chunk)

#         print(f'[INFO] Unzipping data from {url}')
#         with zipfile.ZipFile(file_path, 'r') as zip:
#             zip.extractall(extract_path)
        
#         print(f'[INFO] Removing {file_path}')
#         os.remove(file_path)

#         train_dir = extract_path / 'train'
#         test_dir = extract_path / 'test'
        
#     except ValueError as e:
#         print(e)
#     except requests.exceptions.RequestException as e:
#         print(f"[ERROR] A network error occurred: {e}")
#     except zipfile.BadZipFile:
#         print(f"[ERROR] Failed to unzip file. It may be corrupted or not a valid zip file.")
#     except Exception as e:
#         print(f"[ERROR] An unexpected error occurred: {e}")

#     return train_dir, test_dir

# if __name__ == "__main__":
#     get_data(url=URL, data_path=DATA_PATH)

        