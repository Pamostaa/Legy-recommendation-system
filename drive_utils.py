import os
import logging
import gdown
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Drive folder ID
DRIVE_FOLDER_ID = "1tv04gSLLwMN9agFL14H3cZf9ClC-IOea"

def ensure_dir(directory):
    """Ensure a directory exists, create if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def download_from_drive(folder_id=DRIVE_FOLDER_ID, output_dir="models"):
    """
    Download all files from the specified Google Drive folder.
    
    Args:
        folder_id (str): Google Drive folder ID
        output_dir (str): Local directory to save files
    """
    ensure_dir(output_dir)
    
    try:
        # Download the entire folder
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url=url, output=output_dir, quiet=False)
        logger.info(f"Successfully downloaded files to {output_dir}")
        
        # Verify the downloaded files
        bert_model_dir = os.path.join(output_dir, "bert_preference_model")
        vectors_file = os.path.join(output_dir, "user_vectors.pkl")
        
        if not os.path.exists(bert_model_dir):
            raise FileNotFoundError(f"BERT model directory not found at {bert_model_dir}")
        if not os.path.exists(vectors_file):
            raise FileNotFoundError(f"User vectors file not found at {vectors_file}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error downloading from Drive: {str(e)}")
        raise

def ensure_model_files(force_download=False):
    """
    Ensure model files are present, download if needed.
    
    Args:
        force_download (bool): If True, download files even if they exist locally
    
    Returns:
        bool: True if files are ready to use
    """
    model_dir = "models"
    bert_model_dir = os.path.join(model_dir, "bert_preference_model")
    vectors_file = os.path.join(model_dir, "user_vectors.pkl")
    
    # Check if files already exist
    files_exist = (
        os.path.exists(bert_model_dir) and
        os.path.exists(vectors_file)
    )
    
    if files_exist and not force_download:
        logger.info("Model files already exist locally")
        return True
        
    logger.info("Downloading model files from Google Drive...")
    return download_from_drive() 