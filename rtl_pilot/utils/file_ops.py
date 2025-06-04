"""
File Operations Module

Provides utilities for file management, path operations, and temporary file handling
used throughout the RTL verification workflow.
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Union, Iterator
from contextlib import contextmanager
import json
import yaml

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations for RTL verification workflow"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize FileManager with optional base directory"""
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def read_file(self, file_path: Union[str, Path]) -> str:
        """Read content from a file"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.base_dir / path
            
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise
    
    def write_file(self, file_path: Union[str, Path], content: str, 
                   create_dirs: bool = True) -> None:
        """Write content to a file"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.base_dir / path
            
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"File written successfully: {path}")
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            raise
    
    def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> None:
        """Copy a file from source to destination"""
        try:
            src_path = Path(src)
            dst_path = Path(dst)
            
            if not src_path.is_absolute():
                src_path = self.base_dir / src_path
            if not dst_path.is_absolute():
                dst_path = self.base_dir / dst_path
            
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            
            logger.info(f"File copied: {src_path} -> {dst_path}")
        except Exception as e:
            logger.error(f"Failed to copy file {src} to {dst}: {e}")
            raise
    
    def find_files(self, pattern: str, directory: Optional[Union[str, Path]] = None) -> List[Path]:
        """Find files matching a pattern"""
        try:
            search_dir = Path(directory) if directory else self.base_dir
            if not search_dir.is_absolute():
                search_dir = self.base_dir / search_dir
            
            return list(search_dir.glob(pattern))
        except Exception as e:
            logger.error(f"Failed to find files with pattern {pattern}: {e}")
            return []
    
    def create_directory(self, dir_path: Union[str, Path], parents: bool = True) -> Path:
        """Create a directory"""
        try:
            path = Path(dir_path)
            if not path.is_absolute():
                path = self.base_dir / path
            
            path.mkdir(parents=parents, exist_ok=True)
            logger.info(f"Directory created: {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            raise
    
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """Delete a file"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.base_dir / path
            
            if path.exists():
                path.unlink()
                logger.info(f"File deleted: {path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    def delete_directory(self, dir_path: Union[str, Path], force: bool = False) -> bool:
        """Delete a directory"""
        try:
            path = Path(dir_path)
            if not path.is_absolute():
                path = self.base_dir / path
            
            if path.exists() and path.is_dir():
                if force:
                    shutil.rmtree(path)
                else:
                    path.rmdir()  # Only removes empty directories
                logger.info(f"Directory deleted: {path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete directory {dir_path}: {e}")
            return False
    
    def read_json(self, file_path: Union[str, Path]) -> dict:
        """Read JSON file"""
        try:
            content = self.read_file(file_path)
            return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to read JSON file {file_path}: {e}")
            raise
    
    def write_json(self, file_path: Union[str, Path], data: dict, 
                   indent: int = 2) -> None:
        """Write data to JSON file"""
        try:
            content = json.dumps(data, indent=indent, ensure_ascii=False)
            self.write_file(file_path, content)
        except Exception as e:
            logger.error(f"Failed to write JSON file {file_path}: {e}")
            raise
    
    def read_yaml(self, file_path: Union[str, Path]) -> dict:
        """Read YAML file"""
        try:
            content = self.read_file(file_path)
            return yaml.safe_load(content)
        except Exception as e:
            logger.error(f"Failed to read YAML file {file_path}: {e}")
            raise
    
    def write_yaml(self, file_path: Union[str, Path], data: dict) -> None:
        """Write data to YAML file"""
        try:
            content = yaml.dump(data, default_flow_style=False, 
                              allow_unicode=True, indent=2)
            self.write_file(file_path, content)
        except Exception as e:
            logger.error(f"Failed to write YAML file {file_path}: {e}")
            raise


class TempFileManager:
    """Manages temporary files and directories for RTL verification"""
    
    def __init__(self, prefix: str = "rtl_pilot_", cleanup_on_exit: bool = True):
        """Initialize TempFileManager"""
        self.prefix = prefix
        self.cleanup_on_exit = cleanup_on_exit
        self.temp_items: List[Path] = []
    
    @contextmanager
    def temp_file(self, suffix: str = "", delete: bool = True) -> Iterator[Path]:
        """Create a temporary file"""
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=self.prefix, 
                suffix=suffix, 
                text=True
            )
            os.close(fd)  # Close the file descriptor
            
            temp_file_path = Path(temp_path)
            if not delete:
                self.temp_items.append(temp_file_path)
            
            yield temp_file_path
        
        finally:
            if delete and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")
    
    @contextmanager
    def temp_directory(self, delete: bool = True) -> Iterator[Path]:
        """Create a temporary directory"""
        try:
            temp_dir = tempfile.mkdtemp(prefix=self.prefix)
            temp_dir_path = Path(temp_dir)
            
            if not delete:
                self.temp_items.append(temp_dir_path)
            
            yield temp_dir_path
        
        finally:
            if delete and temp_dir_path.exists():
                try:
                    shutil.rmtree(temp_dir_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp directory {temp_dir_path}: {e}")
    
    def create_temp_file(self, content: str = "", suffix: str = "") -> Path:
        """Create a persistent temporary file"""
        fd, temp_path = tempfile.mkstemp(
            prefix=self.prefix, 
            suffix=suffix, 
            text=True
        )
        
        temp_file_path = Path(temp_path)
        self.temp_items.append(temp_file_path)
        
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to write to temp file {temp_file_path}: {e}")
            raise
        
        return temp_file_path
    
    def create_temp_directory(self) -> Path:
        """Create a persistent temporary directory"""
        temp_dir = tempfile.mkdtemp(prefix=self.prefix)
        temp_dir_path = Path(temp_dir)
        self.temp_items.append(temp_dir_path)
        return temp_dir_path
    
    def cleanup(self) -> None:
        """Clean up all managed temporary items"""
        for item in self.temp_items:
            try:
                if item.exists():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    logger.debug(f"Cleaned up temp item: {item}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp item {item}: {e}")
        
        self.temp_items.clear()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.cleanup_on_exit:
            self.cleanup()


def ensure_file_extension(file_path: Union[str, Path], extension: str) -> Path:
    """Ensure a file path has the specified extension"""
    path = Path(file_path)
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    if path.suffix != extension:
        path = path.with_suffix(extension)
    
    return path


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes"""
    try:
        return Path(file_path).stat().st_size
    except Exception as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        return 0


def is_verilog_file(file_path: Union[str, Path]) -> bool:
    """Check if file is a Verilog file based on extension"""
    verilog_extensions = {'.v', '.vh', '.sv', '.svh', '.verilog'}
    return Path(file_path).suffix.lower() in verilog_extensions


def is_vhdl_file(file_path: Union[str, Path]) -> bool:
    """Check if file is a VHDL file based on extension"""
    vhdl_extensions = {'.vhd', '.vhdl'}
    return Path(file_path).suffix.lower() in vhdl_extensions


def backup_file(file_path: Union[str, Path], backup_suffix: str = ".bak") -> Path:
    """Create a backup of a file"""
    original_path = Path(file_path)
    backup_path = original_path.with_suffix(original_path.suffix + backup_suffix)
    
    try:
        shutil.copy2(original_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup of {file_path}: {e}")
        raise

# Convenience functions for direct usage
_default_manager = FileManager()

def read_file(file_path: Union[str, Path]) -> str:
    """Convenience function to read a file."""
    return _default_manager.read_file(file_path)

def write_file(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> bool:
    """Convenience function to write a file."""
    return _default_manager.write_file(file_path, content, encoding)

def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """Convenience function to copy a file."""
    return _default_manager.copy_file(src, dst)

def ensure_directory(dir_path: Union[str, Path]) -> bool:
    """Convenience function to ensure a directory exists."""
    return _default_manager.ensure_directory(dir_path)

def list_files(directory: Union[str, Path], pattern: str = "*", recursive: bool = False) -> List[Path]:
    """Convenience function to list files."""
    return _default_manager.list_files(directory, pattern, recursive)
