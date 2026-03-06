#!/usr/bin/env python3
"""
Ollama Fine-Tuning Pipeline for Llama3:latest
This script replaces OpenAI with Ollama for fine-tuning a Llama3 model on your articles dataset.
"""
import concurrent.futures
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any
from datasets import Dataset
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
import requests
import os
import sys
from pathlib import Path
from loguru import logger

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ArticleData(BaseModel):
    """Data model for article entries."""
    id: str
    content: str
    platform: str
    author_id: str
    author_full_name: str
    link: str


class TrainingExample(BaseModel):
    """Training example for fine-tuning."""
    instruction: str
    input_text: str = ""
    output: str


class InstructionAnswerSet:
    """Class to handle instruction-answer pairs for training."""
    
    def __init__(self, pairs: List[Tuple[str, str]]):
        """
        Initialize with instruction-answer pairs.
        
        Args:
            pairs: List of tuples containing (instruction, answer) pairs
        """
        self.pairs = pairs
    
    @classmethod
    def from_json(cls, json_str: str) -> 'InstructionAnswerSet':
        """
        Create InstructionAnswerSet from JSON string.
        
        Args:
            json_str: JSON string containing instruction_answer_pairs
            
        Returns:
            InstructionAnswerSet instance
        """
        data = json.loads(json_str)
        pairs = [(pair['instruction'], pair['answer'])
                for pair in data['instruction_answer_pairs']]
        return cls(pairs)
    
    def __iter__(self):
        """Make the class iterable."""
        return iter(self.pairs)


class OllamaConfig(BaseModel):
    """Configuration for Ollama connection."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3:latest"
    timeout: int = 300


class OllamaFineTuningPipeline:
    """Pipeline for fine-tuning Llama3 using Ollama."""
    
    def __init__(self, config: OllamaConfig = None):
        """Initialize the fine-tuning pipeline."""
        self.config = config or OllamaConfig()
        self.articles_dataset = None
        self.training_data = []
        
    def load_articles_from_json(self, file_path: str) -> Dataset:
        """
        Load articles from JSON file and create Hugging Face dataset.
        
        Args:
            file_path: Path to the JSON file containing articles
            
        Returns:
            Hugging Face Dataset with article data
        """
        logger.info(f"📖 Loading articles from {file_path}")
        
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            
            # Handle both direct array and nested artifact_data structures
            articles = data if isinstance(data, list) else data.get("artifact_data", [])
            
            dataset = Dataset.from_dict({
                "id": [item["id"] for item in articles],
                "content": [item["content"] for item in articles],
                "platform": [item["platform"] for item in articles],
                "author_id": [item["author_id"] for item in articles],
                "author_full_name": [item["author_full_name"] for item in articles],
                "link": [item["link"] for item in articles],
            })
            
            logger.info(f"✅ Loaded {len(dataset)} articles")
            self.articles_dataset = dataset
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to load articles: {e}")
            raise
        
    def clean_text(self, text):
        text = re.sub(r"[^\w\s.,!?']", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def extract_substrings(self, dataset: Dataset, min_length: int = 50,
    max_length: int = 2000):
        extracts = []
        sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
        for i, article in enumerate(dataset["content"]):
            # Handle dictionary structure
            if isinstance(article, dict):
                article_text = article.get("Content", "")
            else:
                article_text = str(article)
            
            logger.info(f"Article {i}: {article_text[:100]}...")
            cleaned_article = self.clean_text(article_text)
            
            if len(cleaned_article) < min_length:
                logger.info(f"Article {i} too short after cleaning: {len(cleaned_article)} chars")
                continue
                
            logger.info(f"Article {i} cleaned length: {len(cleaned_article)} chars")
            
            # If article is short enough, use it as is
            if len(cleaned_article) <= max_length:
                extracts.append(cleaned_article)
            else:
                # Split into sentences and create chunks
                sentences = re.split(sentence_pattern, cleaned_article)
                current_chunk = ""
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(current_chunk) + len(sentence) <= max_length:
                        current_chunk += sentence + " "
                    else:
                        if len(current_chunk) >= min_length:
                            extracts.append(current_chunk.strip())
                        current_chunk = sentence + " "
                if len(current_chunk) >= min_length:
                    extracts.append(current_chunk.strip())
        
        logger.info(f"Extracted {len(extracts)} text chunks")
        return extracts
    
    def generate_training_examples(self, num_examples: int = None) -> List[TrainingExample]:
        """
        Generate training examples from articles for fine-tuning.
        
        Args:
            num_examples: Number of examples to generate (default: all articles)
            
        Returns:
            List of training examples
        """
        if self.articles_dataset is None:
            raise ValueError("No articles loaded. Call load_articles_from_json first.")
        
        logger.info("🔄 Generating training examples...")
        
        if num_examples is None:
            num_examples = len(self.articles_dataset)
        
        training_examples = []
        
        for i in tqdm(range(min(num_examples, len(self.articles_dataset))), desc="Creating examples"):
            article = self.articles_dataset[i]
            
            # Generate different types of training examples
            examples = self._create_varied_examples(article)
            training_examples.extend(examples)
        
        # Shuffle the training data
        random.shuffle(training_examples)
        
        logger.info(f"✅ Generated {len(training_examples)} training examples")
        self.training_data = training_examples
        return training_examples
    
    def _create_varied_examples(self, article: Dict[str, Any]) -> List[TrainingExample]:
        """Create varied training examples from a single article."""
        examples = []
        content = article['content']
        title = self._extract_title(content)
        platform = article['platform']
        author = article['author_full_name']
        
        # Example 1: Summarization
        if len(content) > 200:
            examples.append(TrainingExample(
                instruction=f"Summarize the following {platform} article by {author}:",
                input_text=content[:1000],
                output=self._generate_summary(content)
            ))
        
        # Example 2: Question Answering
        qa_pairs = self._generate_qa_pairs(content, title)
        for qa in qa_pairs:
            examples.append(TrainingExample(
                instruction=qa['question'],
                input_text="",
                output=qa['answer']
            ))
        
        # Example 3: Content Analysis
        examples.append(TrainingExample(
            instruction=f"Analyze the key themes and topics in this {platform} content:",
            input_text=content[:800],
            output=self._analyze_content(content)
        ))
        
        # Example 4: Title Generation
        examples.append(TrainingExample(
            instruction="Generate an engaging title for this content:",
            input_text=content[:500],
            output=title if title else f"Insights from {platform}"
        ))
        
        return examples
    
    def _extract_title(self, content: str) -> str:
        """Extract title from content."""
        lines = content.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) < 100 and not line.startswith('#'):
                return line
        return ""
    
    def _generate_summary(self, content: str) -> str:
        """Generate a simple summary of the content."""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Take first few sentences as summary
        summary_sentences = sentences[:3]
        return ". ".join(summary_sentences) + "."
    
    def generate_instruction_answer_pairs(self, extract: str) -> List[Tuple[str, str]]:
        """
        Generate instruction-answer pairs from text extract using Ollama Llama3:latest.
        
        Args:
            extract: Text extract to generate pairs from
            
        Returns:
            List of instruction-answer tuples
        """
        # Try to use Ollama Llama3:latest
        try:
            prompt = f"""Based on the following extract, generate five instruction-answer pairs. Each instruction must ask to write about a specific topic contained in the context. Each answer must provide a relevant paragraph based on information found in the context. Only use concepts from the context to generate instructions. Instructions must never explicitly mention a context, a system, a course, or an extract. Instructions must be self-contained and general. Answers must imitate the writing style of the context. Example instruction: Explain the concept of an LLM Twin. Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice. It's designed to write just like you by incorporating these elements into a language model. The idea is to create a digital replica of your writing habits using advanced AI techniques. Provide your response in JSON format with the following structure:
{{
"instruction_answer_pairs": [
{{"instruction": "...", "answer": "..."}},
...
]
}}
Extract:
{extract}"""

            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json={
                    "model": "llama3:latest",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 1200
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result_text = response.json().get('response', '')
                
                # Try to parse JSON response
                try:
                    # Extract JSON from response if it contains extra text
                    json_start = result_text.find('{')
                    json_end = result_text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = result_text[json_start:json_end]
                        data = json.loads(json_str)
                        pairs = [(pair['instruction'], pair['answer']) 
                                for pair in data['instruction_answer_pairs']]
                        
                        logger.info(f"✅ Generated {len(pairs)} instruction-answer pairs using Ollama Llama3")
                        return pairs
                    else:
                        logger.warning("No JSON found in Ollama response")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from Ollama response: {e}")
                    logger.warning(f"Response text: {result_text[:200]}...")
            else:
                logger.warning(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Failed to generate pairs with Ollama: {e}")
        
        # Fallback to simple pattern-based generation
        logger.info("Using fallback method to generate instruction-answer pairs")
        return self._generate_fallback_pairs(extract)
    
    def _generate_fallback_pairs(self, extract: str) -> List[Tuple[str, str]]:
        """
        Fallback method to generate instruction-answer pairs without OpenAI.
        
        Args:
            extract: Text extract to generate pairs from
            
        Returns:
            List of instruction-answer tuples
        """
        pairs = []
        
        # Simple pattern-based generation
        sentences = re.split(r'[.!?]+', extract)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) >= 2:
            # Generate pairs based on content
            content_lower = extract.lower()
            
            # Pair 1: General topic explanation
            if 'python' in content_lower or 'code' in content_lower:
                pairs.append((
                    "Explain the key programming concepts discussed in the content.",
                    self._generate_summary(extract)
                ))
            
            # Pair 2: Tutorial-style instruction
            if 'tutorial' in content_lower or 'guide' in content_lower or 'how to' in content_lower:
                pairs.append((
                    "Provide a step-by-step approach to the main topic covered.",
                    "The content outlines a systematic approach to understanding the core concepts. It begins with fundamental principles and gradually builds up to more advanced techniques, ensuring comprehensive coverage of the subject matter."
                ))
            
            # Pair 3: Analysis instruction
            pairs.append((
                "Analyze the main themes and insights presented.",
                f"The content explores key themes with practical insights. It provides valuable information on the subject, offering readers a deeper understanding of the concepts discussed through clear explanations and relevant examples."
            ))
            
            # Pair 4: Application instruction
            pairs.append((
                "How can the concepts be applied in practice?",
                "The concepts can be practically applied by following the guidelines and examples provided. The content demonstrates real-world applications that help bridge theory with practice, making the knowledge immediately useful."
            ))
            
            # Pair 5: Summary instruction
            pairs.append((
                "Summarize the key takeaways from the content.",
                self._generate_summary(extract)
            ))
        
        # If no pairs generated, create generic ones
        if not pairs:
            pairs.append((
                "What are the main points discussed in the content?",
                self._generate_summary(extract)
            ))
        
        logger.info(f"✅ Generated {len(pairs)} fallback instruction-answer pairs")
        return pairs
    def create_instruction_dataset(self, dataset: Dataset, num_workers: int = 4) -> Dataset:
        """
        Create instruction dataset from articles using parallel processing.
        
        Args:
            dataset: Hugging Face dataset with articles
            num_workers: Number of parallel workers
            
        Returns:
            Dataset with instruction-output pairs
        """
        extracts = self.extract_substrings(dataset)
        instruction_answer_pairs = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.generate_instruction_answer_pairs, extract)
                     for extract in extracts]
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), desc="Generating pairs"):
                instruction_answer_pairs.extend(future.result())
        
        if instruction_answer_pairs:
            instructions, answers = zip(*instruction_answer_pairs)
            return Dataset.from_dict(
                {"instruction": list(instructions), "output": list(answers)}
            )
        else:
            return Dataset.from_dict({"instruction": [], "output": []})
    
    


def main(dataset_id: str = None):
    """Main function to create and export instruction dataset."""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    logger.info("🤖 Ollama Fine-Tuning Pipeline for Llama3")
    
    # Check command line arguments
    if len(sys.argv) < 2:
        logger.error("Usage: python ollama_fine_tuning_pipeline.py <json_file_path> [dataset_id]")
        logger.error("Example: python ollama_fine_tuning_pipeline.py data/articles.json mlabonne/llmtwin")
        return
    
    json_file_path = sys.argv[1]
    dataset_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if file exists
    if not Path(json_file_path).exists():
        logger.error(f"❌ File not found: {json_file_path}")
        return
    
    try:
        # Initialize pipeline
        pipeline = OllamaFineTuningPipeline()
        
        # 1. Load the raw data
        logger.info("📖 Loading raw data...")
        raw_dataset = pipeline.load_articles_from_json(json_file_path)
        logger.info("Raw dataset:")
        print(raw_dataset.to_pandas())
        
        # 2. Create instruction dataset using Ollama Llama3
        logger.info("🔄 Creating instruction dataset with Ollama Llama3...")
        instruction_dataset = pipeline.create_instruction_dataset(raw_dataset, num_workers=4)
        logger.info("Instruction dataset:")
        print(instruction_dataset.to_pandas())
        
        # 3. Train/test split
        logger.info("� Creating train/test split...")
        filtered_dataset = instruction_dataset.train_test_split(test_size=0.1)
        
        # Display dataset info
        logger.info(f"✅ Dataset created with {len(instruction_dataset)} examples")
        logger.info(f"📊 Train size: {len(filtered_dataset['train'])}")
        logger.info(f"🧪 Test size: {len(filtered_dataset['test'])}")
        
        # 4. Export to hub if dataset_id provided
        if dataset_id:
            logger.info(f"📤 Pushing dataset to hub: {dataset_id}")
            try:
                filtered_dataset.push_to_hub(dataset_id, private=False)
                logger.info(f"✅ Dataset successfully pushed to: https://huggingface.co/datasets/{dataset_id}")
            except Exception as e:
                logger.error(f"❌ Failed to push to hub: {e}")
                logger.info("💡 Make sure you're logged in with: huggingface-cli login")
        else:
            # Save as simple JSON file
            output_file = "data/instruction_pairs.json"
            instruction_pairs = []
            
            # Extract pairs from train and test splits
            for split_name in ['train', 'test']:
                if split_name in filtered_dataset:
                    split_data = filtered_dataset[split_name].to_pandas()
                    for _, row in split_data.iterrows():
                        instruction_pairs.append({
                            'instruction': row['instruction'],
                            'output': row['output']
                        })
            
            # Save as JSON
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(instruction_pairs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Instruction pairs saved as JSON to: {output_file}")
            logger.info(f"📊 Total pairs saved: {len(instruction_pairs)}")
            
            # Also save Hugging Face format for compatibility
            hf_output_file = "instruction_dataset.json"
            filtered_dataset.save_to_disk(hf_output_file)
            logger.info(f"💾 Hugging Face dataset also saved to: {hf_output_file}")
        
        logger.info("🎉 Pipeline completed successfully!")
        return filtered_dataset
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    


if __name__ == "__main__":
    main()
