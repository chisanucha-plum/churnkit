"""Helper functions for the Customer Churn Prediction System.

This module provides utility functions for:
- Safe dictionary access
- Data formatting
- List operations
- Type conversions
- Common calculations
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get value from dictionary with default fallback.
    
    Args:
        data: Dictionary to access
        key: Key to retrieve
        default: Default value if key not found
        
    Returns:
        Value from dictionary or default
    """
    try:
        return data.get(key, default)
    except (AttributeError, TypeError):
        logger.warning(f"Error accessing key '{key}' from data")
        return default


def safe_get_nested(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Safely get nested value from dictionary.
    
    Args:
        data: Dictionary to access
        keys: List of keys for nested access (e.g., ["user", "profile", "email"])
        default: Default value if path not found
        
    Returns:
        Value from nested dictionary or default
    """
    try:
        current = data
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return default
        return current if current is not None else default
    except (AttributeError, TypeError, KeyError):
        logger.warning(f"Error accessing nested keys {keys} from data")
        return default


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string.
    
    Args:
        value: Value to format (0-1 range)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    try:
        return f"{value * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        logger.warning(f"Error formatting percentage: {value}")
        return "N/A"


def format_currency(value: float, currency: str = "$", decimals: int = 2) -> str:
    """Format value as currency string.
    
    Args:
        value: Value to format
        currency: Currency symbol
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    try:
        return f"{currency}{value:,.{decimals}f}"
    except (TypeError, ValueError):
        logger.warning(f"Error formatting currency: {value}")
        return "N/A"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousands separator.
    
    Args:
        value: Value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    try:
        return f"{value:,.{decimals}f}"
    except (TypeError, ValueError):
        logger.warning(f"Error formatting number: {value}")
        return "N/A"


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten nested list.
    
    Args:
        nested_list: Nested list to flatten
        
    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def remove_duplicates(lst: List[Any], preserve_order: bool = True) -> List[Any]:
    """Remove duplicates from list.
    
    Args:
        lst: List to deduplicate
        preserve_order: Whether to preserve original order
        
    Returns:
        List without duplicates
    """
    if preserve_order:
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    else:
        return list(set(lst))


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


def invert_dict(d: Dict[str, Any]) -> Dict[Any, str]:
    """Invert dictionary (swap keys and values).
    
    Args:
        d: Dictionary to invert
        
    Returns:
        Inverted dictionary
    """
    return {v: k for k, v in d.items()}


def filter_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Filter dictionary to only include specified keys.
    
    Args:
        d: Dictionary to filter
        keys: Keys to include
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in d.items() if k in keys}


def exclude_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Exclude specified keys from dictionary.
    
    Args:
        d: Dictionary to filter
        keys: Keys to exclude
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in d.items() if k not in keys}


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    try:
        if old_value == 0:
            return 0.0 if new_value == 0 else float('inf')
        return ((new_value - old_value) / abs(old_value)) * 100
    except (TypeError, ValueError):
        logger.warning(f"Error calculating percentage change: {old_value} -> {new_value}")
        return 0.0


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0-1 range.
    
    Args:
        value: Value to normalize
        min_val: Minimum value in range
        max_val: Maximum value in range
        
    Returns:
        Normalized value (0-1)
    """
    try:
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    except (TypeError, ValueError):
        logger.warning(f"Error normalizing value: {value}")
        return 0.0


def denormalize_value(normalized: float, min_val: float, max_val: float) -> float:
    """Denormalize value from 0-1 range.
    
    Args:
        normalized: Normalized value (0-1)
        min_val: Minimum value in original range
        max_val: Maximum value in original range
        
    Returns:
        Denormalized value
    """
    try:
        return normalized * (max_val - min_val) + min_val
    except (TypeError, ValueError):
        logger.warning(f"Error denormalizing value: {normalized}")
        return 0.0


def get_top_n(items: List[tuple], n: int = 10, reverse: bool = True) -> List[tuple]:
    """Get top N items from list of tuples.
    
    Args:
        items: List of (key, value) tuples
        n: Number of items to return
        reverse: Whether to sort in descending order
        
    Returns:
        Top N items
    """
    try:
        sorted_items = sorted(items, key=lambda x: x[1], reverse=reverse)
        return sorted_items[:n]
    except (TypeError, ValueError):
        logger.warning(f"Error getting top {n} items")
        return []


def convert_to_numeric(value: Any, default: float = 0.0) -> float:
    """Convert value to numeric type.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Numeric value
    """
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
        return default
    except (TypeError, ValueError):
        logger.warning(f"Error converting to numeric: {value}")
        return default


def convert_to_bool(value: Any, default: bool = False) -> bool:
    """Convert value to boolean.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Boolean value
    """
    try:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "yes", "1", "on"}
        if isinstance(value, (int, float)):
            return bool(value)
        return default
    except (TypeError, ValueError):
        logger.warning(f"Error converting to bool: {value}")
        return default


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_dataframe_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics for DataFrame.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with summary statistics
    """
    try:
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isna().sum().to_dict(),
            "missing_percentage": (df.isna().sum() / len(df) * 100).to_dict(),
            "numeric_summary": df.describe().to_dict(),
        }
    except Exception as e:
        logger.error(f"Error getting DataFrame summary: {str(e)}")
        return {}
