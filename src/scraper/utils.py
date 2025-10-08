import time
import logging
from typing import Callable, Any, Optional
from urllib.parse import urlparse
import re
from functools import wraps

logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """Decorator to retry a function on failure with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.2f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed. Last error: {e}")
            raise last_exception
        return wrapper
    return decorator

def is_valid_url(url: str) -> bool:
    """Validate if a URL is properly formatted"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs belong to the same domain"""
    try:
        return urlparse(url1).netloc == urlparse(url2).netloc
    except Exception:
        return False

def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def create_directory_if_not_exists(directory: str) -> None:
    """Create directory if it doesn't exist"""
    import os
    os.makedirs(directory, exist_ok=True)

def get_file_size_mb(filepath: str) -> Optional[float]:
    """Get file size in MB"""
    try:
        import os
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except Exception:
        return None
