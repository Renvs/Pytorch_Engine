import torch
import torch.nn as nn
import requests
import zipfile
import os

from pathlib import Path

def data_from_url(url: str, data_path: str, save_path: str, filename: str = None):
    try:
        if save_path is not None:
            print(f'[INFO] Checking Directory at {save_path}')
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

        if data_path is not None and filename is not None:
            print(f'[INFO] Checking Directory at {data_path}')
            data_path = Path(data_path)
            data_path.mkdir(parents=True, exist_ok=True)
            file_path = data_path / filename

            if url is not None:
                with open(file_path.with_suffix('.zip'), 'wb') as f:        
                    print(f'[INFO] Downloading from {url} at {file_path}')
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

    return train_dir, test_dir

def save_models(model: nn.Module, save_path: Path, model_name: str):

    target_path = Path(save_path)
    target_path.mkdir(parents=True, exist_ok=True)

    model_path = target_path / f'{model_name}.pt'

    try:

        torch.save(model.state_dict(), model_path)
        print(f'[INFO] Model saved at {model_path} / {model_name}.pt')

    except Exception as e:
        
        print(f"[ERROR] Failed to save model: {e}")