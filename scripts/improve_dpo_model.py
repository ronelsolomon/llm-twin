#!/usr/bin/env python3
"""
Script to improve the DPO model with better training data and approaches
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

class DPOModelImprover:
    def __init__(self):
        self.base_model_path = "./dpo_llm_twin_merged"
        self.improved_model_path = "./dpo_llm_twin_improved"
        
    def analyze_current_model(self):
        """Analyze why the current model is performing poorly"""
        print("🔍 Analyzing current model performance...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        
        # Test with different prompts
        test_prompts = [
            "What are your main areas of expertise?",
            "Tell me about yourself.",
            "What programming languages do you know?",
            "Describe your experience with software development."
        ]
        
        print("\n📊 Testing current model responses:")
        print("=" * 60)
        
        for prompt in test_prompts:
            # Try without template
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
            print("-" * 40)
    
    def create_better_dataset(self):
        """Create a better preference dataset with more diverse examples"""
        print("\n📝 Creating improved preference dataset...")
        
        # More diverse and specific preference data
        improved_data = {
            "preference_triples": [
                {
                    "instruction": "What are your main areas of expertise and experience?",
                    "generated_answer": "I have experience in software development and programming.",
                    "extracted_answer": "As Ronel Solomon, I specialize in full-stack software development with expertise in Python, JavaScript, and cloud technologies. I have extensive experience building web applications, working with various databases, and implementing machine learning solutions. My background includes both frontend and backend development, API design, and DevOps practices."
                },
                {
                    "instruction": "Tell me about your programming background.",
                    "generated_answer": "I know how to code.",
                    "extracted_answer": "As Ronel Solomon, I have a strong foundation in computer science with over 5 years of professional programming experience. I'm proficient in multiple programming languages including Python, JavaScript, TypeScript, and Java. I've worked on various projects ranging from web applications to data analysis systems, and I'm passionate about clean code, software architecture, and continuous learning."
                },
                {
                    "instruction": "What technologies do you work with?",
                    "generated_answer": "I use various technologies.",
                    "extracted_answer": "As Ronel Solomon, I work with a comprehensive tech stack including React, Node.js, Python, PostgreSQL, MongoDB, Docker, and cloud platforms like AWS and Google Cloud. I'm experienced with modern development tools, version control systems, and CI/CD pipelines. I stay current with emerging technologies and best practices in software development."
                },
                {
                    "instruction": "Describe your approach to problem-solving.",
                    "generated_answer": "I solve problems by thinking.",
                    "extracted_answer": "As Ronel Solomon, I approach problem-solving systematically by first understanding the requirements, breaking down complex issues into manageable components, researching best practices, and implementing iterative solutions. I believe in data-driven decision making, collaborative problem-solving, and continuous improvement. I document my solutions and seek feedback to ensure robust and maintainable code."
                },
                {
                    "instruction": "What makes you unique as a developer?",
                    "generated_answer": "I am different.",
                    "extracted_answer": "As Ronel Solomon, what makes me unique is my combination of technical expertise and strong communication skills. I excel at translating complex technical concepts into understandable terms for stakeholders, mentoring junior developers, and bridging the gap between technical and business requirements. I'm passionate about creating user-centric solutions and contributing to open-source projects."
                },
                {
                    "instruction": "How do you approach learning new technologies?",
                    "generated_answer": "I learn new things.",
                    "extracted_answer": "As Ronel Solomon, I approach learning new technologies through hands-on experimentation, building projects, reading documentation, and engaging with developer communities. I believe in practical application, staying curious, and maintaining a growth mindset. I regularly contribute to tech blogs, attend workshops, and collaborate with peers to stay current with industry trends."
                },
                {
                    "instruction": "What's your experience with team collaboration?",
                    "generated_answer": "I work with teams.",
                    "extracted_answer": "As Ronel Solomon, I have extensive experience working in agile teams, participating in code reviews, pair programming, and cross-functional collaboration. I value clear communication, constructive feedback, and knowledge sharing. I've led development teams, mentored junior developers, and contributed to establishing coding standards and best practices."
                },
                {
                    "instruction": "How do you ensure code quality?",
                    "generated_answer": "I write good code.",
                    "extracted_answer": "As Ronel Solomon, I ensure code quality through comprehensive testing, code reviews, static analysis, and adherence to coding standards. I implement unit tests, integration tests, and use tools like ESLint, Prettier, and SonarQube. I believe in writing clean, maintainable code with proper documentation and following SOLID principles."
                },
                {
                    "instruction": "What's your experience with databases?",
                    "generated_answer": "I know databases.",
                    "extracted_answer": "As Ronel Solomon, I have extensive experience with both SQL and NoSQL databases including PostgreSQL, MySQL, MongoDB, and Redis. I'm skilled in database design, query optimization, indexing strategies, and data modeling. I've worked with complex data relationships, implemented caching strategies, and designed scalable database architectures."
                },
                {
                    "instruction": "How do you handle project deadlines?",
                    "generated_answer": "I meet deadlines.",
                    "extracted_answer": "As Ronel Solomon, I handle project deadlines through careful planning, prioritization, and realistic estimation. I break down large tasks into smaller milestones, communicate progress transparently, and adjust plans when needed. I believe in delivering quality work on time while maintaining work-life balance and avoiding burnout."
                }
            ]
        }
        
        # Save improved dataset
        output_path = "data/improved_preference_dataset.json"
        with open(output_path, 'w') as f:
            json.dump(improved_data, f, indent=2)
        
        print(f"✅ Improved dataset saved to: {output_path}")
        print(f"📊 Created {len(improved_data['preference_triples'])} preference triples")
        
        return output_path
    
    def test_base_model_directly(self):
        """Test the original DialoGPT model with better prompting"""
        print("\n🧪 Testing original DialoGPT model with improved prompting...")
        
        # Load original DialoGPT
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test with conversational context
        conversational_prompts = [
            "Hello! I'd like to introduce myself. I'm Ronel Solomon, a software developer with expertise in Python and JavaScript. What would you like to know about my experience?",
            "As Ronel Solomon, I have extensive experience in full-stack development. I've worked with React, Node.js, and various cloud technologies. How can I help you today?",
            "I'm Ronel Solomon, and I specialize in building web applications and implementing machine learning solutions. What technical challenges are you facing?"
        ]
        
        print("\n💬 Testing conversational approach:")
        print("=" * 60)
        
        for prompt in conversational_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
            print("-" * 40)
    
    def create_simple_inference_script(self):
        """Create a simpler inference script that works better with the current model"""
        print("\n🔧 Creating improved inference script...")
        
        script_content = '''#!/usr/bin/env python3
"""
Simple inference script for the DPO model with better prompting
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleDPOInference:
    def __init__(self, model_path="./dpo_llm_twin_merged"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model from: {self.model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("✅ Model loaded successfully!")
        
    def generate_response(self, prompt, max_new_tokens=150):
        """Generate response with better prompting"""
        
        # Use conversational context instead of instruction format
        conversational_prompt = f"As Ronel Solomon, {prompt}"
        
        inputs = self.tokenizer(conversational_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        if conversational_prompt in response:
            response = response.replace(conversational_prompt, "").strip()
        
        return response
    
    def chat(self):
        """Simple chat interface"""
        self.load_model()
        
        print("\\n💬 Chat with Ronel's AI Twin!")
        print("Type 'quit' to exit\\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
                
            try:
                response = self.generate_response(user_input)
                print(f"Ronel AI: {response}\\n")
            except Exception as e:
                print(f"Error: {e}\\n")

if __name__ == "__main__":
    inference = SimpleDPOInference()
    inference.chat()
'''
        
        script_path = "scripts/simple_dpo_chat.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"✅ Simple inference script created: {script_path}")
        
        # Make it executable
        Path(script_path).chmod(0o755)

def main():
    """Run all improvement steps"""
    improver = DPOModelImprover()
    
    print("🚀 DPO Model Improvement Pipeline")
    print("=" * 50)
    
    # Step 1: Analyze current model
    improver.analyze_current_model()
    
    # Step 2: Create better dataset
    dataset_path = improver.create_better_dataset()
    
    # Step 3: Test base model
    improver.test_base_model_directly()
    
    # Step 4: Create simple inference script
    improver.create_simple_inference_script()
    
    print("\n🎉 Model improvement analysis complete!")
    print("\n📝 Next steps:")
    print(f"1. Retrain with: python scripts/dpo_fine_tuning_mac.py --config configs/improved_dpo_config.yaml")
    print("2. Try the simple chat: python scripts/simple_dpo_chat.py")
    print("3. Generate more training data to improve responses")

if __name__ == "__main__":
    main()
