"""
Crawlers module for various web platforms.
"""

from .base import BaseCrawler
from .selenium_base import BaseSeleniumCrawler

__all__ = ['BaseCrawler', 'BaseSeleniumCrawler']
