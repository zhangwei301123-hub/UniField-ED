import os
import logging
import sys

def setup_logger(save_dir, log_name="train.log"):
    os.makedirs(save_dir, exist_ok=True)
    
    logger = logging.getLogger("PTv3_Training")
    logger.setLevel(logging.INFO)
    

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 文件 Handler
        file_handler = logging.FileHandler(os.path.join(save_dir, log_name))
        file_handler.setFormatter(formatter)
        
        # 控制台 Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
    return logger