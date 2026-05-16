"""
Flexible DataLoader module for loading data from the data folder.

Usage:
    from src.agentic.data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.load_data('train.csv')
    
    # Or use from config
    df = loader.load_from_config('train')
    
    # With advanced options
    df = loader.load_data('train.csv', columns=['col1', 'col2'], nrows=1000)
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import pandas as pd
import yaml


class DataLoader:
    """
    Flexible data loader for loading CSV files from the data directory.
    
    Attributes:
        data_dir (Path): Path to the data directory
        config (Dict): Configuration dictionary loaded from config.yaml
    """
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None, 
                 config_path: Optional[Union[str, Path]] = None):
        """ 
        Initialize DataLoader.  
        
        Args:
            data_dir: Path to data directory. If None, uses './data' relative to project root
            config_path: Path to config.yaml. If None, uses './config/config.yaml'
        """
        # Determine project root
        if data_dir is None:
            # Try to find project root by looking for pyproject.toml or README.md
            current = Path(__file__).resolve()
            while current != current.parent:
                if (current / 'pyproject.toml').exists() or (current / 'README.md').exists():
                    self.project_root = current
                    break
                current = current.parent
            else:
                # Fallback to current working directory
                self.project_root = Path.cwd()
            
            self.data_dir = self.project_root / 'data'
        else:
            self.data_dir = Path(data_dir)
        
        # Load configuration
        if config_path is None:
            config_path = self.project_root / 'config' / 'config.yaml'
        
        self.config = self._load_config(config_path)
        
        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
                return {}
        return {}
    
    def load_data(self, 
                  filename: str,
                  columns: Optional[List[str]] = None,
                  nrows: Optional[int] = None,
                  skip_rows: Optional[int] = None,
                  dtype: Optional[Dict[str, type]] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            filename: Name of the CSV file (e.g., 'train.csv')
            columns: List of specific columns to load. If None, loads all columns
            nrows: Maximum number of rows to load
            skip_rows: Number of rows to skip at the beginning
            dtype: Dictionary specifying data types for columns
            **kwargs: Additional arguments to pass to pd.read_csv()
        
        Returns:
            pd.DataFrame: Loaded dataframe
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            df = pd.read_csv(
                file_path,
                usecols=columns,
                nrows=nrows,
                skiprows=skip_rows,
                dtype=dtype,
                encoding='utf-8',
                **kwargs
            )
            print(f"[OK] Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading {filename}: {str(e)}")
    
    def load_from_config(self, 
                        key: str,
                        columns: Optional[List[str]] = None,
                        nrows: Optional[int] = None,
                        **kwargs) -> pd.DataFrame:
        """
        Load data using a key from config.yaml.
        
        Args:
            key: Key from config file (e.g., 'train', 'test', 'laws')
            columns: List of specific columns to load
            nrows: Maximum number of rows to load
            **kwargs: Additional arguments to pass to load_data()
        
        Returns:
            pd.DataFrame: Loaded dataframe
            
        Raises:
            KeyError: If key not found in config
        """
        if 'data_dir' not in self.config:
            raise KeyError("'data_dir' section not found in config.yaml")
        
        if key not in self.config['data_dir']:
            available_keys = list(self.config['data_dir'].keys())
            raise KeyError(f"Key '{key}' not found in config. Available keys: {available_keys}")
        
        file_path = self.config['data_dir'][key]
        return self.load_data(file_path, columns=columns, nrows=nrows, **kwargs)
    
    def list_available_files(self) -> List[str]:
        """Get list of all CSV files in the data directory."""
        files = sorted([f.name for f in self.data_dir.glob('*.csv')])
        return files
    
    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """
        Get information about a file without loading it completely.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            Dictionary with file info (size, rows, columns)
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Get row and column count (read just header)
        df_sample = pd.read_csv(file_path, nrows=0)
        
        return {
            'filename': filename,
            'path': str(file_path),
            'size_mb': round(file_size_mb, 2),
            'columns': list(df_sample.columns),
            'num_columns': len(df_sample.columns)
        }
    
    def load_multiple(self, 
                     filenames: List[str],
                     **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Load multiple files at once.
        
        Args:
            filenames: List of filenames to load
            **kwargs: Additional arguments to pass to load_data()
        
        Returns:
            Dictionary with filename as key and dataframe as value
        """
        data = {}
        for filename in filenames:
            try:
                data[filename] = self.load_data(filename, **kwargs)
            except Exception as e:
                print(f"[ERROR] Loading {filename}: {str(e)}")
        return data
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of available data files."""
        files = self.list_available_files()
        summary = {
            'data_dir': str(self.data_dir),
            'available_files': files,
            'config_keys': list(self.config.get('data_dir', {}).keys()),
            'file_details': {}
        }
        
        for filename in files:
            try:
                summary['file_details'][filename] = self.get_file_info(filename)
            except Exception as e:
                print(f"Could not get info for {filename}: {e}")
        
        return summary


# Convenience function for quick loading
def load(filename: str = None, **kwargs) -> pd.DataFrame:
    """
    Quick convenience function to load data without instantiating DataLoader.
    
    Args:
        filename: Name of file to load (e.g., 'train.csv')
        **kwargs: Additional arguments to pass to DataLoader.load_data()
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if filename is None:
        raise ValueError("filename parameter is required")
    
    loader = DataLoader()
    return loader.load_data(filename, **kwargs)


def load_from_config(key: str, **kwargs) -> pd.DataFrame:
    """
    Quick convenience function to load data from config.
    
    Args:
        key: Key from config file (e.g., 'train', 'test')
        **kwargs: Additional arguments to pass to DataLoader.load_from_config()
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    loader = DataLoader()
    return loader.load_from_config(key, **kwargs)
