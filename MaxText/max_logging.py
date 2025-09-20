"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Enhanced logging utilities that support both console output and GCS bucket storage"""

import os
import threading
import time
from datetime import datetime
from typing import Optional, List
import jax
import max_utils


class BucketLogger:
    """Logger that can batch upload logs to GCS bucket"""
    
    def __init__(self, run_name: Optional[str] = None, bucket_dir: Optional[str] = None, upload_interval: int = 60, max_buffer_size: int = 1000):
          self.bucket_dir = bucket_dir
          self.upload_interval = upload_interval  # seconds
          self.max_buffer_size = max_buffer_size
          self.log_buffer: List[str] = []
          self.buffer_lock = threading.Lock()
          self.last_upload_time = time.time()
          self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
          self.run_name = run_name
          
          # Only process 0 should upload logs to avoid duplicates
          self.should_upload = jax.process_index() == 0

          self.local_filename = f"{self.run_name}.{self.session_id}.txt"
          self.backup_dir = f"{self.bucket_dir.rstrip('/')}"

          self.destination_path = f"{self.backup_dir}/{self.local_filename}"
          
          if self.bucket_dir and self.should_upload:
               self._start_upload_thread()

    def backup_main_py_files(self):
        """Backup main py files to GCS bucket"""

        main_filenames = ['exp.py', 'train.py', 'optimizers.py', 'input_pipeline/_pile_data_processing.py', 'configs/base.yml', 'layers/models.py', 'layers/linears.py', 'layers/attentions.py', 'layers/normalizations.py', 'layers/accelerator.py', 'layers/fusion.py', 'layers/mudd.py', 'layers/dc.py', 'layers/mtp.py']
        for filename in main_filenames:
          filename = f'MaxText/{filename}'
          destination_path = os.path.join(self.backup_dir, filename)
          max_utils.upload_blob(destination_path, filename)
          print(f'Backup {filename} to {destination_path}.')
    
    def _start_upload_thread(self):
        """Start background thread for periodic uploads"""
        def upload_worker():
          # upload experiment's model py files
          self.backup_main_py_files()
          while True:
            time.sleep(self.upload_interval)
            self._upload_buffered_logs()
        
        upload_thread = threading.Thread(target=upload_worker, daemon=True)
        upload_thread.start()
    
    def _upload_buffered_logs(self):
        """Upload buffered logs to GCS bucket"""
        if not self.bucket_dir or not self.should_upload:
            return
        
        with self.buffer_lock:
            if not self.log_buffer:
                return
            
            # Create log content
            log_content = '\n'.join(self.log_buffer)
            self.log_buffer.clear()
        
        try:
            # Import here to avoid circular imports
            with open(self.local_filename, 'a+') as f:
                f.write(log_content)
            # Upload to bucket
            max_utils.upload_blob(self.destination_path, self.local_filename)
            print(f"[BucketLogger] Uploaded logs to {self.destination_path}", flush=True)
            # Clean up temp file
          #   os.remove(local_filename)
            
        except Exception as e:
            print(f"[BucketLogger] Failed to upload logs: {str(e)}", flush=True)
    
    def log_to_bucket(self, message: str):
        """Add message to buffer for bucket upload"""
        if not self.bucket_dir or not self.should_upload:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        with self.buffer_lock:
            self.log_buffer.append(formatted_message)
            
            # Upload if buffer is full or enough time has passed
            should_upload = (len(self.log_buffer) >= self.max_buffer_size or 
                           time.time() - self.last_upload_time >= self.upload_interval)
            
            if should_upload:
                self.last_upload_time = time.time()
                # Schedule upload in background
                threading.Thread(target=self._upload_buffered_logs, daemon=True).start()


# Global logger instance
_bucket_logger: Optional[BucketLogger] = None


def initialize_bucket_logging(bucket_dir: Optional[str] = None, upload_interval: int = 60, max_buffer_size: int = 1000, run_name: Optional[str] = None):
    """Initialize bucket logging with specified parameters"""
    global _bucket_logger
    if bucket_dir:
        _bucket_logger = BucketLogger(run_name, bucket_dir, upload_interval, max_buffer_size)
        print(f"[MaxText] Initialized bucket logging to {bucket_dir}", flush=True)


def log(user_str, debug=True, save_to_bucket=True):
    """Enhanced logging function that supports both console and bucket output"""
    if debug:
        print(user_str, flush=True)
    
    # Also save to bucket if enabled
    if save_to_bucket and _bucket_logger:
        _bucket_logger.log_to_bucket(user_str)
