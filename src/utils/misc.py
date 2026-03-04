"""
Miscellaneous utility functions
"""
from typing import Iterator, List, Any


def batch(iterable: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """
    Split a list into batches of specified size
    
    Args:
        iterable: List to split into batches
        batch_size: Size of each batch
        
    Yields:
        Lists of size batch_size (except possibly the last one)
    """
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx:min(ndx + batch_size, length)]
