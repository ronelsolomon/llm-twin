#!/usr/bin/env python3
"""
Convert LinkedIn text to JSON format for the instruction dataset pipeline.
"""
import json
from pathlib import Path

def convert_linkedin_to_json():
    """Convert LinkedIn text file to JSON format compatible with the pipeline."""
    
    # Read the LinkedIn text file
    linkedin_file = Path("data/linkedin_raw_text.txt")
    if not linkedin_file.exists():
        print(f"❌ File not found: {linkedin_file}")
        return
    
    with open(linkedin_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into meaningful chunks by looking for natural breaks
    chunks = []
    
    # Try splitting by double newlines or major section headers
    sections = content.split('\n\n')
    
    for section in sections:
        section = section.strip()
        if len(section) > 100:  # Only include substantial sections
            # Further split very long sections
            if len(section) > 1000:
                words = section.split()
                current_chunk = ""
                for word in words:
                    current_chunk += word + " "
                    if len(current_chunk) >= 800:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(section)
    
    # Create JSON structure
    articles = []
    for i, chunk in enumerate(chunks):
        article = {
            "id": f"linkedin_chunk_{i}",
            "content": {
                "Title": f"Professional Profile Section {i+1}",
                "Content": chunk,
                "language": "en"
            },
            "link": "https://www.linkedin.com/in/ronel-solomon",
            "platform": "linkedin",
            "author_id": "ronel_solomon",
            "author_full_name": "Ronel Solomon",
            "created_at": "2026-03-05T15:12:00.000000",
            "updated_at": "2026-03-05T15:12:00.000000"
        }
        articles.append(article)
    
    # Save to JSON file
    output_file = Path("data/linkedin_articles.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(articles)} chunks to {output_file}")
    print(f"📄 Sample chunk previews:")
    for i, article in enumerate(articles[:3]):  # Show first 3
        content_preview = article['content']['Content'][:150] + "..." if len(article['content']['Content']) > 150 else article['content']['Content']
        print(f"   Chunk {i+1}: {content_preview}")

if __name__ == "__main__":
    convert_linkedin_to_json()
