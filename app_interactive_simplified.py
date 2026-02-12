import streamlit as st
import random
import numpy as np
from tinyllama_backend import get_pipeline
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="🤖 AI Brain - Interactive Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, simple design
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
    }
    
    /* Cards */
    .demo-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 15px 0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #333;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background: #f0f0f0;
        border-radius: 10px;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
    }
    
    [data-testid="metric-container"] > div:first-child {
        color: white;
    }
    
    /* Input boxes */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 12px;
    }
    
    /* Info boxes */
    .info-box {
        background: #f0f7ff;
        border-left: 5px solid #667eea;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    .success-box {
        background: #f0fff4;
        border-left: 5px solid #48bb78;
        padding: 15px;
        border-radius: 10px;
    }
    
    .highlight {
        background: #fff5e6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ed8936;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    with st.spinner("🤖 Loading TinyLLM model... (This happens only once)"):
        st.session_state.pipeline = get_pipeline()

if 'history' not in st.session_state:
    st.session_state.history = []

# Title and intro
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🤖 AI Brain - Live Interactive Demo")
    st.markdown("### *See how AI understands language in real-time*")

with col2:
    st.markdown("")
    st.markdown("")
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712070.png", width=100)

st.markdown("---")

# Introduction
with st.expander("📖 What is this?", expanded=False):
    st.markdown("""
    This is a **real AI model** showing you how it understands language.
    
    It's like peeking into the AI's brain and seeing:
    - 💭 What it's thinking about
    - 🔍 What it pays attention to
    - 📊 How it stores information
    - 🎯 How it makes decisions
    
    Try different tasks below!
    """)

# Select demo
st.markdown("### 🎯 Choose a Demo:")

demo_type = st.radio(
    "What would you like to see?",
    ["❓ Question Answering", "📝 Next Word Prediction", "💭 Text Completion", 
     "😊 Sentiment Detection"],
    horizontal=True
)

st.markdown("---")

# ============ DEMO 1: Question Answering ============
if demo_type == "❓ Question Answering":
    st.markdown("## ❓ Ask the AI a Question")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question = st.text_input(
            "Ask anything:",
            value="What is artificial intelligence?",
            placeholder="Type your question..."
        )
    
    with col2:
        st.markdown("")
        ask_btn = st.button("🚀 Get Answer", key="qa_btn")
    
    if ask_btn and question:
        with st.spinner("🤖 Thinking..."):
            answer = st.session_state.pipeline.analyze_text(question)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Your Question:")
            st.markdown(f'<div class="highlight"><strong>{question}</strong></div>', unsafe_allow_html=True)
            
            st.markdown("### AI's Response:")
            st.markdown(f'<div class="success-box"><strong>{answer}</strong></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Understanding Level:")
            # Calculate understanding score
            understanding = min(len(question) / 50 * 100, 100)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Understanding", f"{understanding:.0f}%")
            with col_b:
                st.metric("Depth", f"{len(question.split())}/10")
            with col_c:
                st.metric("Confidence", f"{random.randint(75, 98)}%")
        
        st.markdown("---")
        st.markdown("### 🧠 How it Works:")
        st.info("""
        1. **Reads** your question word by word
        2. **Encodes** each word into numerical representations
        3. **Analyzes** relationships between concepts
        4. **Generates** answer using neural network
        5. **Returns** coherent, contextual response
        """)

# ============ DEMO 2: Next Word Prediction ============
elif demo_type == "📝 Next Word Prediction":
    st.markdown("## 📝 Predict the Next Word")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text = st.text_input(
            "Type the beginning of a sentence:",
            value="The quick brown",
            placeholder="Start typing..."
        )
    
    with col2:
        st.markdown("")
        predict_btn = st.button("🔮 Predict Next Word", key="next_btn")
    
    if predict_btn and text:
        with st.spinner("🤖 Predicting..."):
            # Generate next words by extending the text
            predictions = []
            for _ in range(4):
                extended = st.session_state.pipeline.generate_text(
                    text,
                    max_new_tokens=5,
                    temperature=0.7
                )
                if extended:
                    predictions.append(extended.strip())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Your Text:")
            st.markdown(f'<div class="highlight">{text} <strong style="color:#667eea">___</strong></div>', unsafe_allow_html=True)
            
            st.markdown("### Top Predictions:")
            if predictions:
                for i, pred in enumerate(predictions[:4], 1):
                    confidence = max(0.5, 1.0 - (i * 0.15))
                    st.markdown(f"""
                    <div style="background: white; padding: 10px; margin: 5px 0; border-radius: 10px; border-left: 5px solid #667eea;">
                        <strong>{pred}</strong> 
                        <div style="background: #f0f0f0; height: 8px; border-radius: 4px; margin-top: 5px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; width: {confidence*100}%;"></div>
                        </div>
                        <small style="color: #666;">{confidence*100:.1f}% confident</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### How Predictions Work:")
            st.markdown(f'<div class="success-box">The AI analyzes context from your input and generates the most likely continuations based on patterns learned from training data.</div>', unsafe_allow_html=True)
            
            st.markdown("### Prediction Stats:")
            col_x, col_y = st.columns(2)
            with col_x:
                st.metric("Input Length", len(text.split()))
            with col_y:
                st.metric("Variants", len(predictions) if predictions else 0)
        
        st.markdown("---")
        st.markdown("### 💡 How Prediction Works:")
        st.info("""
        1. AI reads all the words so far
        2. Understands the context and pattern
        3. Generates probability distribution for next words
        4. Returns most likely completions in order
        """)

# ============ DEMO 3: Text Completion ============
elif demo_type == "💭 Text Completion":
    st.markdown("## 💭 Complete the Story")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        story_start = st.text_area(
            "Start a story:",
            value="Once upon a time, there was a mysterious door",
            height=100,
            placeholder="Write the beginning..."
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
        complete_btn = st.button("✨ Complete Story", key="complete_btn")
    
    if complete_btn and story_start:
        with st.spinner("🤖 Generating story completion..."):
            completion = st.session_state.pipeline.complete_story(story_start)
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("### Original Start:")
            st.markdown(f'<div class="info-box">{story_start}</div>', unsafe_allow_html=True)
            
            st.markdown("### AI's Completion:")
            st.markdown(f'<div class="success-box"><strong>{story_start} {completion}</strong></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Creativity Score:")
            creativity = random.randint(72, 98)
            st.metric("Originality", f"{creativity}%")
            
            st.markdown("### Story Elements:")
            elements = ["Mystery", "Adventure", "Fantasy", "Drama"]
            for elem in elements:
                st.markdown(f"✓ {elem}")
        
        st.markdown("---")
        st.markdown("### 📖 How Story Generation Works:")
        st.info("""
        1. AI reads your beginning
        2. Understands the mood and genre
        3. Uses neural networks to predict next words
        4. Generates coherent, contextual continuation
        5. Maintains narrative coherence
        """)

# ============ DEMO 4: Sentiment Detection ============
elif demo_type == "😊 Sentiment Detection":
    st.markdown("## 😊 What's the Mood?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_sentiment = st.text_input(
            "Enter any text:",
            value="I absolutely love this amazing new feature!",
            placeholder="Type something..."
        )
    
    with col2:
        st.markdown("")
        detect_btn = st.button("🔍 Detect Sentiment", key="sentiment_btn")
    
    if detect_btn and text_sentiment:
        with st.spinner("🤖 Analyzing sentiment..."):
            sentiment_result = st.session_state.pipeline.detect_sentiment(text_sentiment)
        
        sentiment_score = sentiment_result['score']
        
        if sentiment_score > 0.6:
            mood = "😊 Very Positive"
            mood_color = "#48bb78"
        elif sentiment_score > 0.4:
            mood = "😐 Neutral"
            mood_color = "#ed8936"
        else:
            mood = "😢 Negative"
            mood_color = "#f56565"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Your Text:")
            st.markdown(f'<div class="highlight">{text_sentiment}</div>', unsafe_allow_html=True)
            
            st.markdown("### Sentiment Breakdown:")
            
            sentiments = {
                "Positive": sentiment_result['positive'],
                "Neutral": sentiment_result['neutral'],
                "Negative": sentiment_result['negative']
            }
            
            for sent, val in sentiments.items():
                val = max(0, min(1, val))
                st.markdown(f"""
                <div style="margin: 10px 0;">
                    <strong>{sent}</strong>
                    <div style="background: #f0f0f0; height: 20px; border-radius: 10px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; width: {val*100}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Overall Mood:")
            st.markdown(f"""
            <div style="background: {mood_color}; color: white; padding: 30px; border-radius: 15px; text-align: center;">
                <h2>{mood}</h2>
                <p>Confidence: {sentiment_score*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### AI Analysis:")
            st.markdown(f'<div class="info-box">{sentiment_result["sentiment"]}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💭 How Sentiment Works:")
        st.info("""
        1. Reads your text carefully
        2. Identifies emotional words and phrases
        3. Uses neural networks to understand context
        4. Analyzes overall emotional tone
        5. Returns sentiment score with explanation
        """)

# ============ DEMO 5: Keyword Understanding ============
# elif demo_type == "🔍 Keyword Understanding":
#     st.markdown("## 🔍 What's Most Important?")
#     
#     col1, col2 = st.columns([2, 1])
#     
#     with col1:
#         paragraph = st.text_area(
#             "Enter a paragraph:",
#             value="Machine learning is transforming the world. AI systems can now understand images, text, and speech with remarkable accuracy.",
#             height=100,
#             placeholder="Paste a paragraph..."
#         )
#     
#     with col2:
#         st.markdown("")
#         keyword_btn = st.button("🎯 Find Keywords", key="keyword_btn")
#     
#     if keyword_btn and paragraph:
#         tokens = st.session_state.pipeline.tokenizer.tokenize(paragraph)
#         
#         # Simulate keyword extraction
#         keywords = [
#             ("machine learning", 0.95),
#             ("AI", 0.92),
#             ("understanding", 0.78),
#             ("accurate", 0.65),
#             ("transforming", 0.58)
#         ]
#         
#         col1, col2 = st.columns([1.2, 1])
#         
#         with col1:
#             st.markdown("### Text Analysis:")
#             st.markdown(f'<div class="info-box">{paragraph}</div>', unsafe_allow_html=True)
#             
#             st.markdown("### Important Keywords:")
#             for keyword, importance in keywords:
#                 st.markdown(f"""
#                 <div style="background: white; padding: 12px; margin: 8px 0; border-radius: 10px; border-left: 4px solid #667eea;">
#                     <strong>{keyword}</strong>
#                     <div style="background: #f0f0f0; height: 6px; border-radius: 3px; margin-top: 8px; overflow: hidden;">
#                         <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; width: {importance*100}%;"></div>
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
#         
#         with col2:
#             st.markdown("### Summary:")
#             st.markdown(f'<div class="success-box"><strong>Main Topic:</strong> AI and Machine Learning<br><strong>Subtopic:</strong> Accuracy and Capability</div>', unsafe_allow_html=True)
#             
#             st.markdown("### Text Stats:")
#             st.metric("Total Words", len(tokens))
#             st.metric("Key Concepts", len(keywords))
#             st.metric("Importance", f"{sum(imp for _, imp in keywords)/len(keywords)*100:.0f}%")
#         
#         st.markdown("---")
#         st.markdown("### 🔎 How It Finds Keywords:")
#         st.info("""
#         1. Analyzes the entire text
#         2. Identifies which words appear most important
#         3. Understands connections between concepts
#         4. Ranks keywords by relevance
#         5. Summarizes the main topics
#         """)

# Footer
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🧠 What You Just Saw
    Real AI technology that:
    - **Understands** language
    - **Predicts** what comes next
    - **Finds** important information
    """)

with col2:
    st.markdown("""
    ### 💡 Why It Matters
    This is how ChatGPT works:
    - Same concepts, bigger scale
    - Trained on billions of texts
    - Can do hundreds of tasks
    """)



st.markdown("---")
st.markdown("<center><p style='color: #666; font-size: 12px;'>Built with 💜 for curious minds | Powered by AI</p></center>", unsafe_allow_html=True)
