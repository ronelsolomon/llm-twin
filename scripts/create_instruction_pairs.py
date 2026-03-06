#!/usr/bin/env python3
"""
Create instruction pairs directly from LinkedIn data and save as JSON
"""
import json
from pathlib import Path

def create_instruction_pairs():
    """Create instruction pairs from LinkedIn articles"""
    
    # Load LinkedIn articles
    with open('data/linkedin_articles.json', 'r') as f:
        articles = json.load(f)
    
    # Create instruction pairs manually based on the content
    instruction_pairs = []
    
    for article in articles:
        content = article['content']['Content']
        
        # Create different types of instruction pairs based on content
        if 'ML Engineer' in content or 'Data Science' in content:
            instruction_pairs.append({
                'instruction': 'Describe Ronel Solomon\'s professional background and expertise.',
                'output': content[:500] + '...' if len(content) > 500 else content
            })
        
        if 'University' in content or 'Education' in content:
            instruction_pairs.append({
                'instruction': 'What is Ronel Solomon\'s educational background?',
                'output': content[:400] + '...' if len(content) > 400 else content
            })
        
        if 'skills' in content.lower() or 'programming' in content.lower():
            instruction_pairs.append({
                'instruction': 'What are Ronel Solomon\'s technical skills and programming expertise?',
                'output': content[:450] + '...' if len(content) > 450 else content
            })
        
        if 'internship' in content.lower() or 'experience' in content.lower():
            instruction_pairs.append({
                'instruction': 'Describe Ronel Solomon\'s work experience and internships.',
                'output': content[:500] + '...' if len(content) > 500 else content
            })
    
    # Add some general instruction pairs
    instruction_pairs.extend([
        {
            'instruction': 'Summarize Ronel Solomon\'s professional profile.',
            'output': 'Ronel Solomon is an ML Engineer/AI Video Solutions specialist based in San Francisco Bay Area with expertise in Generative AI, ML pipelines, and production deployment. He holds a Master of Science in Data Science from University of San Francisco and Bachelor\'s degrees from UC Berkeley.'
        },
        {
            'instruction': 'What are Ronel Solomon\'s key technical skills?',
            'output': 'Ronel Solomon has strong programming skills in Python, SQL, Java, JavaScript, C++, and TypeScript. He specializes in Generative AI & LLMs, ML pipeline design, production deployment, and data engineering with experience in AWS, GCP, and Azure cloud platforms.'
        },
        {
            'instruction': 'Describe Ronel Solomon\'s educational achievements.',
            'output': 'Ronel Solomon holds an MS in Data Science from University of San Francisco, Bachelor\'s degrees in Data Science from UC Berkeley, a Certificate in Entrepreneurship & Technology from UC Berkeley, and completed coursework at San Jose City College.'
        }
    ])
    
    # Save to JSON
    output_file = Path('data/instruction_pairs.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(instruction_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created {len(instruction_pairs)} instruction pairs in {output_file}")
    
    # Show examples
    print("\n📄 Sample instruction pairs:")
    for i, pair in enumerate(instruction_pairs[:3]):
        print(f"\n{i+1}. Instruction: {pair['instruction']}")
        print(f"   Output: {pair['output'][:100]}...")

if __name__ == "__main__":
    create_instruction_pairs()
