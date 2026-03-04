#!/usr/bin/env python3
"""
Cache management utility for the standalone data extractor
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.standalone_data_extractor import StandaloneDataExtractor

def main():
    """Cache management utility"""
    extractor = StandaloneDataExtractor()
    
    print("=== Cache Management Utility ===")
    print()
    
    while True:
        print("Options:")
        print("1. Show cache statistics")
        print("2. Clear all cache")
        print("3. Clear GitHub cache only")
        print("4. Clear article cache only")
        print("5. Clear cache older than 24 hours")
        print("6. Clear cache older than 1 week")
        print("7. Exit")
        print()
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == "1":
            stats = extractor.get_cache_stats()
            print("\n📊 Cache Statistics:")
            print(f"   Total entries: {stats['total_entries']}")
            print(f"   GitHub entries: {stats['github_entries']}")
            print(f"   Article entries: {stats['article_entries']}")
            print(f"   Cache size: {stats['cache_size_mb']} MB")
            if stats['oldest_entry']:
                print(f"   Oldest entry: {stats['oldest_entry']}")
            if stats['newest_entry']:
                print(f"   Newest entry: {stats['newest_entry']}")
            print()
            
        elif choice == "2":
            print("\n🧹 Clearing all cache...")
            extractor.clear_cache()
            print()
            
        elif choice == "3":
            print("\n🧹 Clearing GitHub cache...")
            extractor.clear_cache(content_type='github')
            print()
            
        elif choice == "4":
            print("\n🧹 Clearing article cache...")
            extractor.clear_cache(content_type='article')
            print()
            
        elif choice == "5":
            print("\n🧹 Clearing cache older than 24 hours...")
            extractor.clear_cache(max_age_hours=24)
            print()
            
        elif choice == "6":
            print("\n🧹 Clearing cache older than 1 week...")
            extractor.clear_cache(max_age_hours=168)  # 7 days * 24 hours
            print()
            
        elif choice == "7":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.\n")

if __name__ == "__main__":
    main()
