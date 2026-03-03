#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from github import GithubCrawler
from domain.documents import UserDocument

def test_github_crawler():
    # Get an existing user
    user = UserDocument.find(id="ronelsolomon")
    if not user:
        print("User not found!")
        return
    
    print(f"Testing GitHub crawler with user: {user.full_name}")
    
    # Initialize crawler
    crawler = GithubCrawler()
    
    # Test with a sample repository
    test_repo = "https://github.com/microsoft/vscode"
    
    try:
        print(f"Crawling repository: {test_repo}")
        crawler.extract(test_repo, user=user)
        print("✅ Repository crawled successfully!")
    except Exception as e:
        print(f"❌ Error crawling repository: {e}")

if __name__ == "__main__":
    test_github_crawler()
