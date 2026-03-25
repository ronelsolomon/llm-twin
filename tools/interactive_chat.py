#!/usr/bin/env python3
"""
Interactive Chat Interface for LLM Twin
A command-line chat interface using the inference pipeline.
"""
import sys
import readline
import time
import uuid
from pathlib import Path
from typing import List, Dict
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference_pipeline import (
    LLMTwinInferencePipeline,
    InferenceConfig,
    GenerationRequest,
    create_inference_pipeline
)
from loguru import logger


class ChatSession:
    """Interactive chat session with LLM Twin"""
    
    def __init__(self, pipeline: LLMTwinInferencePipeline):
        self.pipeline = pipeline
        self.session_id = str(uuid.uuid4())
        self.history: List[Dict[str, str]] = []
        self.user_id = "interactive_user"
        
    def add_to_history(self, role: str, message: str):
        """Add message to chat history"""
        self.history.append({
            "role": role,
            "message": message,
            "timestamp": time.time()
        })
    
    def get_context_prompt(self, current_prompt: str, context_window: int = 3) -> str:
        """Create context-aware prompt from history"""
        # Get recent messages for context
        recent_history = self.history[-context_window*2:] if self.history else []
        
        context_parts = []
        for msg in recent_history:
            role_prefix = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role_prefix}: {msg['message']}")
        
        # Add current prompt
        context_parts.append(f"User: {current_prompt}")
        context_parts.append("Assistant:")
        
        return "\n".join(context_parts)
    
    def chat(self):
        """Start interactive chat session"""
        print("💬 LLM Twin Interactive Chat")
        print("=" * 50)
        print("Commands:")
        print("  /help     - Show this help")
        print("  /clear    - Clear chat history")
        print("  /history  - Show chat history")
        print("  /save     - Save chat history")
        print("  /config   - Show current configuration")
        print("  /quit     - Exit chat")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Handle commands
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                if not user_input:
                    continue
                
                # Add user message to history
                self.add_to_history("user", user_input)
                
                # Generate response
                print("🤖 Twin: ", end="", flush=True)
                
                # Use context-aware prompt
                context_prompt = self.get_context_prompt(user_input)
                
                request = GenerationRequest(
                    prompt=context_prompt,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    metadata={"chat_mode": True}
                )
                
                response = self.pipeline.generate(request)
                assistant_message = response.generated_text
                
                print(assistant_message)
                
                # Add assistant response to history
                self.add_to_history("assistant", assistant_message)
                
                # Show generation stats (optional)
                if response.generation_time > 2.0:  # Only show for slow generations
                    print(f"⏱️  ({response.generation_time:.1f}s, {response.token_count} tokens)")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def handle_command(self, command: str) -> bool:
        """Handle chat commands"""
        cmd = command.lower().strip()
        
        if cmd == "/help":
            self.show_help()
        elif cmd == "/clear":
            self.history.clear()
            print("🧹 Chat history cleared")
        elif cmd == "/history":
            self.show_history()
        elif cmd == "/save":
            self.save_history()
        elif cmd == "/config":
            self.show_config()
        elif cmd in ["/quit", "/exit"]:
            return False
        else:
            print(f"❌ Unknown command: {command}")
            print("Type /help for available commands")
        
        return True
    
    def show_help(self):
        """Show help information"""
        print("\n💬 Chat Commands:")
        print("  /help     - Show this help")
        print("  /clear    - Clear chat history")
        print("  /history  - Show chat history")
        print("  /save     - Save chat history to file")
        print("  /config   - Show current model configuration")
        print("  /quit     - Exit chat")
        print("\n💡 Tips:")
        print("  - The model remembers recent conversation context")
        print("  - Use /clear to start fresh")
        print("  - Your chat history is automatically saved when you quit")
    
    def show_history(self):
        """Show chat history"""
        if not self.history:
            print("📝 No chat history yet")
            return
        
        print("\n📝 Chat History:")
        print("-" * 40)
        
        for i, msg in enumerate(self.history, 1):
            role_prefix = "You" if msg["role"] == "user" else "Twin"
            print(f"{i}. {role_prefix}: {msg['message'][:100]}{'...' if len(msg['message']) > 100 else ''}")
    
    def show_config(self):
        """Show current model configuration"""
        config = self.pipeline.config
        print(f"\n⚙️  Model Configuration:")
        print(f"  Model path: {config.model_path}")
        print(f"  Device: {config.device}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Max tokens: {config.max_new_tokens}")
        print(f"  Top-p: {config.top_p}")
        print(f"  Top-k: {config.top_k}")
        
        # Get model info
        model_info = self.pipeline.get_model_info()
        print(f"\n📊 Model Info:")
        print(f"  Parameters: {model_info.get('num_parameters', 'N/A'):,}")
        print(f"  Vocab size: {model_info.get('vocab_size', 'N/A'):,}")
    
    def save_history(self):
        """Save chat history to file"""
        if not self.history:
            print("📝 No chat history to save")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump({
                    "session_id": self.session_id,
                    "user_id": self.user_id,
                    "timestamp": timestamp,
                    "history": self.history
                }, f, indent=2)
            
            print(f"💾 Chat history saved to {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save history: {e}")


def main():
    """Main function for interactive chat"""
    import time
    import uuid
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout, 
        level="WARNING",  # Reduce log noise in chat
        format="<level>{message}</level>"
    )
    
    print("🤖 LLM Twin Interactive Chat")
    print("=" * 50)
    
    # Check model availability
    model_path = Path("./dpo_llm_twin_merged")
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please ensure the model is available before starting chat")
        return
    
    try:
        # Create pipeline
        print("🔄 Loading model...")
        pipeline = create_inference_pipeline()
        
        # Check model info
        model_info = pipeline.get_model_info()
        print(f"✅ Model loaded: {model_info.get('num_parameters', 'N/A'):,} parameters")
        print(f"📱 Device: {model_info.get('device', 'N/A')}")
        
        # Start chat session
        chat_session = ChatSession(pipeline)
        chat_session.chat()
        
        # Save history on exit
        if chat_session.history:
            chat_session.save_history()
        
        # Cleanup
        pipeline.unload_model()
        
    except KeyboardInterrupt:
        print("\n👋 Chat interrupted by user")
    except Exception as e:
        print(f"❌ Error starting chat: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
