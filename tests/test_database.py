#!/usr/bin/env python3
"""
Test script to demonstrate the file-based database system
"""

from domain.documents import UserDocument, RepositoryDocument, ArticleDocument

def test_database():
    print("Testing file-based database system...")
    
    # Test UserDocument
    print("\n1. Creating UserDocument...")
    user = UserDocument(id="ronelsolomon", full_name="Ronel Solomon")
    user.save()
    print(f"User created: {user.full_name} (ID: {user.id})")
    
    # Test RepositoryDocument
    print("\n2. Creating RepositoryDocument...")
    repo_content = {
        "README.md": "# Test Repository\nThis is a test repository",
        "main.py": "print('Hello, World!')"
    }
    
    repo = RepositoryDocument(
        content=repo_content,
        name="test-repo",
        link="https://github.com/ronelsolomon/test-repo",
        platform="github",
        author_id=user.id,
        author_full_name=user.full_name
    )
    repo.save()
    print(f"Repository created: {repo.name}")
    
    # Test ArticleDocument
    print("\n3. Creating ArticleDocument...")
    article_content = {
        "Title": "Test Article",
        "Subtitle": "A test article subtitle",
        "Content": "This is the full content of the test article...",
        "language": "en"
    }
    
    article = ArticleDocument(
        content=article_content,
        link="https://example.com/test-article",
        platform="custom",
        author_id=user.id,
        author_full_name=user.full_name
    )
    article.save()
    print(f"Article created: {article.content['Title']}")
    
    # Test finding documents
    print("\n4. Testing find operations...")
    
    found_user = UserDocument.find(id="ronelsolomon")
    print(f"Found user: {found_user.full_name if found_user else 'None'}")
    
    found_repo = RepositoryDocument.find(link="https://github.com/ronelsolomon/test-repo")
    print(f"Found repository: {found_repo.name if found_repo else 'None'}")
    
    found_article = ArticleDocument.find(platform="custom")
    print(f"Found article: {found_article.content['Title'] if found_article else 'None'}")
    
    # Test get all
    print("\n5. Testing get all operations...")
    all_users = UserDocument.get_all()
    print(f"Total users: {len(all_users)}")
    
    all_repos = RepositoryDocument.get_all()
    print(f"Total repositories: {len(all_repos)}")
    
    all_articles = ArticleDocument.get_all()
    print(f"Total articles: {len(all_articles)}")
    
    print("\n✅ Database test completed successfully!")
    print(f"📁 Data saved in: {RepositoryDocument._db_path}")

if __name__ == "__main__":
    test_database()
