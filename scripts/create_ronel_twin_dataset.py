#!/usr/bin/env python3
"""
Create a comprehensive dataset for Ronel's LLM twin using all available data sources.
This script processes profile data, articles, instruction pairs, and other content
to create a dataset that captures Ronel's voice, expertise, and communication style.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import random
from datetime import datetime

class RonelTwinDatasetCreator:
    def __init__(self, data_dir: str = "/Users/ronel/Downloads/llm twin/data"):
        self.data_dir = Path(data_dir)
        self.dataset = []
        
        # Ronel's personal information for context
        self.ronel_context = {
            "name": "Ronel Solomon",
            "email": "ronelsolomon@berkeley.edu",
            "role": "ML Engineer / AI Video Solutions",
            "location": "San Francisco Bay Area",
            "linkedin": "www.linkedin.com/in/ronel-solomon",
            "blog": "medium.com/@ronelsolomon",
            "github": "github.com/ronelsolomon",
            "expertise": [
                "Generative AI & LLMs",
                "Machine Learning Engineering", 
                "AI-powered educational tools",
                "Data pipelines",
                "Cloud platforms (AWS, GCP, Azure)",
                "Python, SQL, Java, JavaScript, C++, TypeScript",
                "Vector search techniques",
                "API deployment",
                "Real-time analytics"
            ]
        }
    
    def load_profile_data(self) -> List[Dict]:
        """Load and process Ronel's profile data."""
        profile_file = self.data_dir / "Profile.md"
        if not profile_file.exists():
            return []
        
        with open(profile_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create conversation pairs from profile
        profile_entries = []
        
        # Split profile into sections
        sections = content.split('\n\n')
        current_section = ""
        
        for section in sections:
            if section.strip():
                current_section += section + "\n\n"
                
                # Create Q&A pairs from profile content
                if any(keyword in section.lower() for keyword in ['experience', 'education', 'skills', 'summary']):
                    profile_entries.append({
                        "instruction": f"Tell me about your {section.split()[0].lower()} background and experience.",
                        "output": current_section.strip(),
                        "context": "professional_profile"
                    })
                    profile_entries.append({
                        "instruction": f"What are your key qualifications and achievements?",
                        "output": current_section.strip(),
                        "context": "professional_profile"
                    })
        
        return profile_entries
    
    def load_articles_content(self) -> List[Dict]:
        """Load and process articles and LinkedIn content."""
        articles = []
        
        # Load articles.json
        articles_file = self.data_dir / "articles.json"
        if articles_file.exists():
            with open(articles_file, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
                
            for article in article_data:
                if isinstance(article, dict) and 'content' in article:
                    content = article['content']
                    if content.get('Content') and content.get('Content') != "MediumSitemap":
                        articles.append({
                            "instruction": f"Tell me about {content.get('Title', 'your work')}",
                            "output": content['Content'],
                            "context": "article_content"
                        })
                        articles.append({
                            "instruction": "What projects have you been working on recently?",
                            "output": content['Content'],
                            "context": "article_content"
                        })
        
        # Load LinkedIn articles
        linkedin_file = self.data_dir / "linkedin_articles.json"
        if linkedin_file.exists():
            with open(linkedin_file, 'r', encoding='utf-8') as f:
                linkedin_data = json.load(f)
                
            for chunk in linkedin_data:
                if isinstance(chunk, dict) and 'content' in chunk:
                    content = chunk['content']
                    if content.get('Content'):
                        articles.append({
                            "instruction": f"Tell me about your experience as a {self.ronel_context['role']}",
                            "output": content['Content'],
                            "context": "linkedin_profile"
                        })
                        articles.append({
                            "instruction": "What's your professional background?",
                            "output": content['Content'],
                            "context": "linkedin_profile"
                        })
        
        return articles
    
    def load_instruction_pairs(self) -> List[Dict]:
        """Load existing instruction pairs and enhance them."""
        pairs_file = self.data_dir / "instruction_pairs.json"
        if not pairs_file.exists():
            return []
        
        with open(pairs_file, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        
        enhanced_pairs = []
        for pair in pairs:
            # Add Ronel's personal touch to existing pairs
            enhanced_pair = pair.copy()
            enhanced_pair["context"] = "ronel_knowledge"
            enhanced_pairs.append(enhanced_pair)
        
        return enhanced_pairs
    
    def create_conversational_entries(self) -> List[Dict]:
        """Create conversational entries that capture Ronel's speaking style."""
        conversational = [
            {
                "instruction": "Hi Ronel! Can you introduce yourself?",
                "output": f"Hi! I'm Ronel Solomon, a Machine Learning Engineer specializing in AI Video Solutions based in the San Francisco Bay Area. I have extensive experience developing AI-powered educational tools, working with generative AI and LLMs, and building scalable ML pipelines. I'm passionate about leveraging cutting-edge technology to solve real-world problems.",
                "context": "introduction"
            },
            {
                "instruction": "What's your background in machine learning?",
                "output": "I've worked across various ML engineering roles, from developing AI-powered digital stethoscopes at MediNai to building Slack and Teams bots using generative AI at Metaphor Data. I've also created comprehensive AI textbook generation systems and worked on real-time analytics dashboards. My expertise spans the full ML pipeline - from data engineering to model deployment.",
                "context": "background"
            },
            {
                "instruction": "What technologies do you work with?",
                "output": "I work extensively with Python, SQL, Java, JavaScript, C++, and TypeScript. On the ML side, I leverage Hugging Face transformers, LangChain, and various LLM frameworks. For cloud infrastructure, I have hands-on experience with AWS, GCP, and Azure, along with tools like Apache Spark, Airflow, and Databricks for data pipelines.",
                "context": "technologies"
            },
            {
                "instruction": "Tell me about your education.",
                "output": "I have a Master of Science in Data Science from the University of San Francisco, and a Bachelor's degree in Data Science from UC Berkeley. I also hold a Certificate in Entrepreneurship & Technology from Berkeley. My academic background has been crucial in shaping my approach to solving complex data science and ML problems.",
                "context": "education"
            },
            {
                "instruction": "What kind of projects do you enjoy working on?",
                "output": "I'm particularly passionate about projects that combine AI with practical applications - like the AI-powered textbook generator I developed, or the digital stethoscope for medical diagnosis. I enjoy building systems that have real impact, whether it's in education, healthcare, or improving business processes through automation and intelligent systems.",
                "context": "projects"
            }
        ]
        
        return conversational
    
    def create_expertise_entries(self) -> List[Dict]:
        """Create entries that showcase Ronel's technical expertise."""
        expertise_entries = [
            {
                "instruction": "How do you approach building ML pipelines?",
                "output": "When building ML pipelines, I focus on scalability and reliability. I typically use Apache Spark and Airflow for data processing, integrate with cloud platforms like AWS or Azure, and ensure proper monitoring and error handling. The key is building robust systems that can handle production workloads while maintaining data quality and model performance.",
                "context": "expertise"
            },
            {
                "instruction": "What's your experience with generative AI?",
                "output": "I've worked extensively with generative AI, from developing Slack and Teams bots using vector search to creating AI-powered content generation systems. I leverage Hugging Face transformers and various LLM APIs to build applications that can understand, summarize, and generate human-like text. The key is understanding how to properly prompt and fine-tune models for specific use cases.",
                "context": "expertise"
            },
            {
                "instruction": "How do you handle model deployment?",
                "output": "For model deployment, I focus on ensuring reliability, scalability, and performance. I've deployed models using Flask/.NET on Azure, built real-time inference APIs, and implemented proper monitoring and versioning. The goal is to create production-ready systems that can handle real-world traffic while maintaining model accuracy and response times.",
                "context": "expertise"
            }
        ]
        
        return expertise_entries
    
    def create_dataset(self) -> List[Dict]:
        """Create the complete dataset by combining all sources."""
        print("Creating Ronel twin dataset...")
        
        # Load all data sources
        profile_entries = self.load_profile_data()
        articles = self.load_articles_content()
        instruction_pairs = self.load_instruction_pairs()
        conversational = self.create_conversational_entries()
        expertise = self.create_expertise_entries()
        
        # Combine all entries
        all_entries = (
            profile_entries + 
            articles + 
            instruction_pairs + 
            conversational + 
            expertise
        )
        
        # Add metadata to each entry
        for entry in all_entries:
            entry["author"] = "Ronel Solomon"
            entry["timestamp"] = datetime.now().isoformat()
            entry["dataset_version"] = "1.0"
        
        print(f"Created {len(all_entries)} training entries")
        return all_entries
    
    def save_dataset(self, output_path: str = None):
        """Save the dataset to a file."""
        if output_path is None:
            output_path = self.data_dir / "ronel_twin_dataset.json"
        
        dataset = self.create_dataset()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {output_path}")
        print(f"Total entries: {len(dataset)}")
        
        # Create a smaller sample for quick testing
        sample_size = min(50, len(dataset))
        sample_dataset = random.sample(dataset, sample_size)
        sample_path = self.data_dir / "ronel_twin_dataset_sample.json"
        
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Sample dataset ({sample_size} entries) saved to {sample_path}")
        
        return dataset

if __name__ == "__main__":
    creator = RonelTwinDatasetCreator()
    dataset = creator.save_dataset()
