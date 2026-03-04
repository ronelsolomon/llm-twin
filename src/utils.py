"""
Utility functions for the LLM Twin project
"""
from typing import Tuple


def split_user_full_name(full_name: str) -> Tuple[str, str]:
    """
    Split a full name into first and last name
    
    Args:
        full_name: The full name to split
        
    Returns:
        Tuple of (first_name, last_name)
    """
    if not full_name:
        return "", ""
    
    name_parts = full_name.strip().split()
    if len(name_parts) == 0:
        return "", ""
    elif len(name_parts) == 1:
        return name_parts[0], ""
    else:
        return name_parts[0], " ".join(name_parts[1:])
