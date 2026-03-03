#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from loguru import logger
from markitdown import MarkItDown
from domain.documents import ArticleDocument, UserDocument
import json
from datetime import datetime

def process_profile_pdf():
    """
    Process Profile.pdf using markitdown and convert to LinkedIn-style data
    """
    logger.info("🔄 Processing Profile.pdf with markitdown...")
    
    # Path to the PDF file
    pdf_path = Path("data/Profile.pdf")
    
    if not pdf_path.exists():
        logger.error(f"❌ Profile.pdf not found at {pdf_path}")
        return False
    
    try:
        # Initialize markitdown
        md = MarkItDown()
        
        # Convert PDF to markdown
        logger.info("📄 Converting PDF to markdown...")
        result = md.convert(str(pdf_path))
        
        markdown_content = result.text_content
        logger.success(f"✅ Successfully converted PDF to markdown ({len(markdown_content)} characters)")
        
        # Save markdown content
        markdown_path = Path("data/Profile.md")
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logger.info(f"💾 Markdown saved to {markdown_path}")
        
        # Process the markdown content into LinkedIn-style posts
        process_markdown_to_linkedin_posts(markdown_content)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing PDF: {e}")
        return False

def process_markdown_to_linkedin_posts(markdown_content):
    """
    Convert markdown content to LinkedIn-style posts and save to database
    """
    logger.info("🔄 Converting markdown to LinkedIn posts...")
    
    # Create user for the posts
    user = UserDocument(
        id="ronel_solomon",
        full_name="Ronel Solomon",
        email="ronel@example.com"
    )
    
    # Split markdown into sections (potential posts)
    sections = split_markdown_into_sections(markdown_content)
    
    logger.info(f"📊 Found {len(sections)} sections to process")
    
    saved_posts = 0
    
    for i, section in enumerate(sections):
        try:
            # Extract title and content from section
            title, content = extract_title_and_content(section)
            
            if not content.strip():
                continue
            
            # Create LinkedIn-style post data
            article_data = {
                "Title": title or f"Professional Update #{i+1}",
                "Subtitle": "From Profile PDF",
                "Content": content,
                "language": "en"
            }
            
            # Generate a fake LinkedIn URL for the post
            post_url = f"https://www.linkedin.com/posts/ronel-solomon-profile-update-{i+1}-{datetime.now().strftime('%Y%m%d')}/"
            
            # Check if already exists
            existing = ArticleDocument.find(link=post_url)
            if existing:
                logger.info(f"⏭️  Post already exists: {title}")
                continue
            
            # Create and save article document
            instance = ArticleDocument(
                content=article_data,
                link=post_url,
                platform="www.linkedin.com",
                author_id=user.id,
                author_full_name=user.full_name,
            )
            instance.save()
            saved_posts += 1
            
            logger.info(f"✅ Saved post {saved_posts}: {title}")
            
        except Exception as e:
            logger.error(f"❌ Error processing section {i+1}: {e}")
    
    logger.success(f"🎉 Successfully saved {saved_posts} LinkedIn posts from Profile PDF!")
    
    # Show summary
    show_linkedin_posts_summary()

def split_markdown_into_sections(markdown_content):
    """
    Split markdown content into logical sections
    """
    sections = []
    
    # Split by double newlines or headers
    lines = markdown_content.split('\n')
    current_section = []
    
    for line in lines:
        # If it's a header or empty line followed by content, start new section
        if line.startswith('#') or (not line.strip() and current_section):
            if current_section:
                sections.append('\n'.join(current_section))
                current_section = []
        
        current_section.append(line)
    
    # Add the last section
    if current_section:
        sections.append('\n'.join(current_section))
    
    # Filter out empty sections and merge very short ones
    filtered_sections = []
    for section in sections:
        if len(section.strip()) > 50:  # Only keep sections with substantial content
            filtered_sections.append(section)
    
    return filtered_sections

def extract_title_and_content(section):
    """
    Extract title and content from a markdown section
    """
    lines = section.split('\n')
    title = ""
    content_lines = []
    
    for line in lines:
        if line.startswith('#') and not title:
            # Extract header as title
            title = line.lstrip('#').strip()
        elif line.strip():
            content_lines.append(line)
    
    content = '\n'.join(content_lines).strip()
    
    # If no title found, use first line as title
    if not title and content_lines:
        title = content_lines[0][:100]
        content = '\n'.join(content_lines[1:]).strip()
    
    return title, content

def show_linkedin_posts_summary():
    """
    Show summary of LinkedIn posts in database
    """
    try:
        # Read all articles
        with open('data/articles.json', 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        linkedin_posts = [a for a in articles if a.get('platform') == 'www.linkedin.com']
        
        logger.info(f"\n📊 LinkedIn Posts Summary:")
        logger.info(f"   Total posts: {len(linkedin_posts)}")
        
        # Show latest 5 posts
        logger.info(f"\n📝 Latest Posts:")
        for i, post in enumerate(linkedin_posts[-5:]):
            title = post.get('content', {}).get('Title', 'No Title')
            content = post.get('content', {}).get('Content', 'No Content')
            logger.info(f"   {i+1}. {title}")
            logger.info(f"      {content[:100]}...")
            logger.info("")
        
    except Exception as e:
        logger.error(f"Error showing summary: {e}")

if __name__ == "__main__":
    logger.info("🚀 Profile PDF to LinkedIn Posts Converter")
    logger.info("=" * 50)
    
    success = process_profile_pdf()
    
    if success:
        logger.success("✨ Processing completed successfully!")
    else:
        logger.error("❌ Processing failed!")
