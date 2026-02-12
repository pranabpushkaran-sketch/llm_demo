"""
Mistral 7B Backend for Interactive AI Demo
Uses transformers library to run Mistral 7B locally
Optimized for Streamlit Cloud deployment
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

class MistralPipeline:
    """Mistral 7B Pipeline for text generation"""
    
    def __init__(self):
        """Initialize Mistral 7B Instruct model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations for cloud
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            load_in_8bit=False  # Use half precision instead for memory efficiency
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
    
    def generate_text(self, prompt, max_length=100, max_new_tokens=None, temperature=0.7, top_p=0.9):
        """Generate text completion"""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Use max_new_tokens if provided, otherwise calculate from max_length
        if max_new_tokens is None:
            max_new_tokens = max(max_length - inputs['input_ids'].shape[1], 10)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the completion (remove the prompt)
        completion = generated_text[len(prompt):].strip()
        
        return completion
    
    def complete_story(self, story_start):
        """Complete a story given the beginning"""
        prompt = f"Story: {story_start}\nContinuation: "
        
        completion = self.generate_text(
            prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.95
        )
        
        return completion.split('\n')[0]  # Get first sentence
    
    def detect_sentiment(self, text):
        """Analyze sentiment of text"""
        prompt = f"Text: {text}\n\nSentiment Analysis:\nThis text expresses a "
        
        completion = self.generate_text(
            prompt,
            max_new_tokens=20,
            temperature=0.5,
            top_p=0.9
        )
        
        # Extract sentiment keywords
        sentiment_lower = completion.lower()
        
        positive_score = 0.5
        
        if any(word in sentiment_lower for word in ['positive', 'happy', 'joy', 'love', 'great', 'amazing', 'wonderful', 'excellent']):
            positive_score = 0.8
        elif any(word in sentiment_lower for word in ['negative', 'sad', 'angry', 'hate', 'terrible', 'awful', 'horrible', 'bad']):
            positive_score = 0.2
        elif any(word in sentiment_lower for word in ['neutral', 'mixed', 'ambiguous']):
            positive_score = 0.5
        
        return {
            'sentiment': completion.strip(),
            'score': positive_score,
            'positive': positive_score,
            'neutral': 1 - abs(positive_score - 0.5) * 2,
            'negative': 1 - positive_score
        }
    
    def analyze_text(self, text):
        """General text analysis"""
        prompt = f"Analyze this text: {text}\n\nAnalysis: "
        
        analysis = self.generate_text(
            prompt,
            max_new_tokens=50,
            temperature=0.6
        )
        
        return analysis.strip()


@st.cache_resource
def load_mistral():
    """Load Mistral 7B model once and cache it"""
    return MistralPipeline()


def get_pipeline():
    """Get or create Mistral pipeline"""
    return load_mistral()
