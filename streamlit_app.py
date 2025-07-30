"""
Streamlit UI for Conversation Token Extractor

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import json
from conversation_extractor import ConversationAnalyzer, extract_tokens
import time
from datetime import datetime
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Conversation Token Extractor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stAlert > div {
        padding-top: 15px;
        padding-bottom: 15px;
    }
    .token-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .user-info {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Create data directory for persistence
DATA_DIR = Path("user_extraction_data")
DATA_DIR.mkdir(exist_ok=True)

# Helper functions for data persistence
def save_user_data(user_id, data):
    """Save user data to JSON file"""
    file_path = DATA_DIR / f"user_{user_id}.json"
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_user_data(user_id):
    """Load user data from JSON file"""
    file_path = DATA_DIR / f"user_{user_id}.json"
    if file_path.exists():
        with open(file_path, 'r') as f:
            return json.load(f)
    return {"user_id": user_id, "extractions": [], "statistics": {}}

def get_all_users():
    """Get list of all users who have saved data"""
    users = []
    for file_path in DATA_DIR.glob("user_*.json"):
        user_id = file_path.stem.replace("user_", "")
        users.append(user_id)
    return sorted(users)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
    st.session_state.current_user_id = None
    st.session_state.user_data = None

# Header
col_title, col_user = st.columns([3, 1])
with col_title:
    st.title("🔍 Conversation Token Extractor")
    st.markdown("Extract emotions, activities, and people from conversations using multiple AI strategies")

with col_user:
    # User ID input
    st.markdown("### 👤 User ID")
    user_id = st.text_input(
        "Enter your ID:",
        placeholder="e.g., user123",
        key="user_id_input",
        help="Enter a unique ID to save your extraction history"
    )
    
    if user_id and user_id != st.session_state.current_user_id:
        st.session_state.current_user_id = user_id
        st.session_state.user_data = load_user_data(user_id)
        st.success(f"✅ Logged in as: **{user_id}**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Strategy Selection
    st.subheader("🎯 Extraction Strategy")
    strategy = st.selectbox(
        "Choose extraction method:",
        ["mixed", "together", "rule_based"],
        index=0,
        format_func=lambda x: {
            "mixed": "🔀 Mixed (Rule-based + API)",
            "together": "🌐 API Only (Cloud-based)",
            "rule_based": "📝 Rule-based Only (Local)"
        }[x],
        help="Select how to extract conversation tokens"
    )
    
    # Strategy descriptions
    strategy_info = {
        "mixed": "🔀 **Mixed Strategy**: Combines rule-based patterns with AI analysis for best accuracy. Requires API key.",
        "together": "🌐 **API Only**: Uses cloud-based AI models for contextual analysis. Requires API key.",
        "rule_based": "📝 **Rule-based Only**: Uses local keyword matching and patterns. No API key needed."
    }
    st.info(strategy_info[strategy])
    
    # API Key Configuration (only show if needed)
    api_key = None
    if strategy in ["mixed", "together"]:
        st.subheader("🔑 API Configuration")
        api_key = st.text_input(
            "Together API Key:",
            type="password",
            placeholder="Enter your Together API key",
            help="Get your API key from https://api.together.xyz/"
        )
    
    # Model selection (only show if using API)
    selected_model = "mistralai/Mistral-7B-Instruct-v0.1"
    if strategy in ["mixed", "together"]:
        model_options = [
            "mistralai/Mistral-7B-Instruct-v0.1",
            # "mistralai/Mistral-7B-Instruct-v0.2",
            # "meta-llama/Llama-2-7b-chat-hf",
            # "meta-llama/Llama-2-13b-chat-hf",
            # "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
        ]
        
        selected_model = st.selectbox(
            "Select Model:",
            model_options,
            index=0,
            help="Choose the language model for conversation analysis"
        )
        
        # Model information
        st.info(f"🤖 Using Model: **{selected_model}**")
    else:
        st.info("🤖 Using: **Local Rule-based Processing**")
    
    # Initialize analyzer based on strategy
    analyzer_ready = False
    
    if strategy == "rule_based":
        # Rule-based doesn't need API key
        if st.session_state.analyzer is None:
            with st.spinner("Initializing rule-based extractor..."):
                try:
                    st.session_state.analyzer = ConversationAnalyzer(strategy="rule_based")
                    st.success("✅ Rule-based extractor ready!")
                    analyzer_ready = True
                except Exception as e:
                    st.error(f"❌ Error initializing rule-based extractor: {str(e)}")
        else:
            analyzer_ready = True
    
    elif strategy in ["mixed", "together"] and api_key:
        # API-based strategies need API key
        if st.session_state.analyzer is None:
            spinner_text = "Initializing Mixed Strategy..." if strategy == "mixed" else "Initializing Together API..."
            with st.spinner(spinner_text):
                try:
                    st.session_state.analyzer = ConversationAnalyzer(
                        strategy=strategy, 
                        api_key=api_key,
                        model_name=selected_model
                    )
                    success_text = f"✅ {strategy.title()} strategy initialized successfully!"
                    st.success(success_text)
                    analyzer_ready = True
                except Exception as e:
                    st.error(f"❌ Error initializing {strategy} strategy: {str(e)}")
                    st.warning("Please check your API key and internet connection.")
        else:
            analyzer_ready = True
    
    elif strategy in ["mixed", "together"] and not api_key:
        st.warning("⚠️ Please enter your Together API key for this strategy.")
    
    # Force reinitialization if strategy changed
    if hasattr(st.session_state, 'current_strategy') and st.session_state.current_strategy != strategy:
        st.session_state.analyzer = None
        st.session_state.current_strategy = strategy
        st.rerun()
    else:
        st.session_state.current_strategy = strategy
    
    st.divider()
    
    # User Management Section
    if st.session_state.current_user_id:
        st.header("👤 User Info")
        st.markdown(f"""
        <div class='user-info'>
            <b>Current User:</b> {st.session_state.current_user_id}<br>
            <b>Total Extractions:</b> {len(st.session_state.user_data.get('extractions', []))}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout"):
            st.session_state.current_user_id = None
            st.session_state.user_data = None
            st.rerun()
    
    # Show existing users
    existing_users = get_all_users()
    if existing_users:
        st.divider()
        st.header("📚 Existing Users")
        selected_user = st.selectbox(
            "Load user data:",
            [""] + existing_users,
            help="Select a user to load their history"
        )
        if selected_user and st.button("📂 Load User"):
            st.session_state.current_user_id = selected_user
            st.session_state.user_data = load_user_data(selected_user)
            st.rerun()

# Main content area
if not st.session_state.current_user_id:
    st.warning("⚠️ Please enter a User ID to start extracting and saving conversations.")
elif strategy in ["mixed", "together"] and not api_key:
    st.warning("⚠️ Please enter your Together API key in the sidebar for this extraction strategy.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("💬 Input Conversation")
        
        # Topic/Context input
        conversation_topic = st.text_input(
            "Topic/Context (optional):",
            placeholder="e.g., Team meeting, Personal chat, Project discussion",
            help="Add a topic or context for this conversation"
        )
        
        # Text input area
        conversation_text = st.text_area(
            "Enter your conversation text:",
            value=st.session_state.get('conversation_text', ''),
            height=200,
            placeholder="Type or paste a conversation here...\n\nExample: I'm excited about meeting Sarah tomorrow! We'll discuss the new project plans.",
            key="conversation_input"
        )
        
        # Extract button
        extract_button = st.button("🚀 Extract Tokens", type="primary", use_container_width=True)

    with col2:
        st.header("📊 Extraction Results")
        
        if extract_button and conversation_text and st.session_state.analyzer:
            analysis_text = {
                "rule_based": "rule-based patterns",
                "together": f"API ({selected_model})",
                "mixed": f"mixed strategy ({selected_model})"
            }
            with st.spinner(f"Analyzing conversation with {analysis_text.get(strategy, strategy)}..."):
                start_time = time.time()
                
                try:
                    # Extract tokens
                    tokens = st.session_state.analyzer.analyze(conversation_text)
                    extraction_time = time.time() - start_time
                    
                    # Create extraction record
                    extraction_record = {
                        'timestamp': datetime.now().isoformat(),
                        'user_id': st.session_state.current_user_id,
                        'topic': conversation_topic or "No topic specified",
                        'text': conversation_text,
                        'tokens': tokens,
                        'extraction_time': extraction_time
                    }
                    
                    # Add to user data
                    if 'extractions' not in st.session_state.user_data:
                        st.session_state.user_data['extractions'] = []
                    st.session_state.user_data['extractions'].append(extraction_record)
                    
                    # Update statistics
                    if 'statistics' not in st.session_state.user_data:
                        st.session_state.user_data['statistics'] = {}
                    
                    stats = st.session_state.user_data['statistics']
                    stats['total_extractions'] = len(st.session_state.user_data['extractions'])
                    stats['total_emotions'] = stats.get('total_emotions', 0) + len(tokens['emotions'])
                    stats['total_activities'] = stats.get('total_activities', 0) + len(tokens['activities'])
                    stats['total_people'] = stats.get('total_people', 0) + len(tokens['people'])
                    
                    # Save to file
                    save_user_data(st.session_state.current_user_id, st.session_state.user_data)
                    
                    # Display metrics
                    col2_1, col2_2, col2_3 = st.columns(3)
                    
                    with col2_1:
                        st.metric("😊 Emotions", len(tokens['emotions']))
                    with col2_2:
                        st.metric("🏃 Activities", len(tokens['activities']))
                    with col2_3:
                        st.metric("👥 People", len(tokens['people']))
                    
                    st.caption(f"⏱️ Extraction time: {extraction_time:.2f}s")
                    
                    # Display tokens
                    st.subheader("Extracted Tokens")
                    
                    # Emotions
                    if tokens['emotions']:
                        st.markdown("**😊 Emotions**")
                        emotion_colors = {
                            'happy': '🟢', 'excited': '🟢', 'content': '🟢',
                            'sad': '🔵', 'anxious': '🟡', 'angry': '🔴',
                            'surprised': '🟣', 'confident': '🟠'
                        }
                        emotions_html = " ".join([
                            f"{emotion_colors.get(e, '⚪')} {e}"
                            for e in tokens['emotions']
                        ])
                        st.markdown(emotions_html)
                    else:
                        st.info("No emotions detected")
                    
                    # Activities
                    if tokens['activities']:
                        st.markdown("**🏃 Activities**")
                        activities_html = " • ".join([f"`{a}`" for a in tokens['activities']])
                        st.markdown(activities_html)
                    else:
                        st.info("No activities detected")
                    
                    # People
                    if tokens['people']:
                        st.markdown("**👥 People**")
                        people_html = " • ".join([
                            f"**{p}**" if p[0].isupper() else f"_{p}_"
                            for p in tokens['people']
                        ])
                        st.markdown(people_html)
                    else:
                        st.info("No people detected")
                    
                    # JSON output (expandable)
                    with st.expander("📋 View JSON Output"):
                        st.json(tokens)
                    
                    st.success("✅ Extraction saved to user history!")
                    
                except Exception as e:
                    st.error(f"❌ Error during extraction: {str(e)}")
        
        elif extract_button and not conversation_text:
            st.warning("⚠️ Please enter some conversation text first!")
        
        elif extract_button and not st.session_state.analyzer:
            st.error("⚠️ API not initialized. Please check your API key and try again.")

# User History Section
if st.session_state.current_user_id and st.session_state.user_data:
    st.divider()
    st.header(f"📜 Extraction History for {st.session_state.current_user_id}")
    
    # User Statistics
    if 'statistics' in st.session_state.user_data:
        stats = st.session_state.user_data['statistics']
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Total Extractions", stats.get('total_extractions', 0))
        with col_stat2:
            st.metric("Total Emotions", stats.get('total_emotions', 0))
        with col_stat3:
            st.metric("Total Activities", stats.get('total_activities', 0))
        with col_stat4:
            st.metric("Total People", stats.get('total_people', 0))
    
    # Filter options
    col_filter1, col_filter2 = st.columns([2, 1])
    with col_filter1:
        topic_filter = st.text_input("🔍 Filter by topic:", placeholder="Enter topic to filter")
    with col_filter2:
        sort_order = st.selectbox("Sort by:", ["Newest First", "Oldest First"])
    
    # Display history
    extractions = st.session_state.user_data.get('extractions', [])
    if extractions:
        # Apply filtering and sorting
        if topic_filter:
            extractions = [e for e in extractions if topic_filter.lower() in e.get('topic', '').lower()]
        
        if sort_order == "Newest First":
            extractions = reversed(extractions)
        
        # Display extractions
        for i, extraction in enumerate(extractions):
            timestamp = datetime.fromisoformat(extraction['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            topic = extraction.get('topic', 'No topic')
            text_preview = extraction['text'][:100] + '...' if len(extraction['text']) > 100 else extraction['text']
            
            with st.expander(f"🕐 {timestamp} | 📌 {topic} | {text_preview}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Topic:** {topic}")
                    st.markdown(f"**Full Text:** {extraction['text']}")
                    st.markdown(f"**Timestamp:** {timestamp}")
                
                with col2:
                    tokens = extraction['tokens']
                    st.metric("Emotions", len(tokens['emotions']))
                    st.metric("Activities", len(tokens['activities']))
                    st.metric("People", len(tokens['people']))
                
                st.json(tokens)
                st.caption(f"Extraction time: {extraction.get('extraction_time', 0):.2f}s")
        
        # Export options
        st.divider()
        col_export1, col_export2 = st.columns([1, 1])
        
        with col_export1:
            if st.button("📥 Download User History (JSON)"):
                st.download_button(
                    label="💾 Download JSON",
                    data=json.dumps(st.session_state.user_data, indent=2, default=str),
                    file_name=f"user_{st.session_state.current_user_id}_history.json",
                    mime="application/json"
                )
        
        with col_export2:
            if st.button("🗑️ Clear User History"):
                if st.checkbox("I'm sure I want to delete all my history"):
                    st.session_state.user_data = {"user_id": st.session_state.current_user_id, "extractions": [], "statistics": {}}
                    save_user_data(st.session_state.current_user_id, st.session_state.user_data)
                    st.rerun()
    else:
        st.info("No extraction history yet. Start analyzing conversations to build your history!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit | 
    Multiple AI Strategies: Rule-based, API, and Mixed</p>
</div>
""", unsafe_allow_html=True)

# Instructions in sidebar
with st.sidebar:
    st.divider()
    st.header("📖 How to Use")
    st.markdown("""
    1. **Choose extraction strategy** (Mixed recommended)
    2. **Enter API key** (if using Mixed or API strategies)
    3. **Enter your User ID** to start tracking
    4. **Enter conversation text** to analyze
    5. **Add a topic** (optional) for organization
    6. **Click Extract Tokens** to analyze
    7. **View results** and history by user
    
    **Strategy Guide:**
    - 🔀 **Mixed**: Best accuracy, uses both methods
    - 🌐 **API Only**: Advanced AI, requires internet
    - 📝 **Rule-based**: Fast, works offline, basic accuracy
    
    **Token Types:**
    - 😊 **Emotions**: Feelings expressed
    - 🏃 **Activities**: Actions mentioned
    - 👥 **People**: Names and pronouns
    
    **Extraction Strategies:**
    - 🔀 **Mixed**: Rule-based + API (best accuracy)
    - 🌐 **API Only**: Cloud-based AI models
    - 📝 **Rule-based**: Local pattern matching
    
    **Data Storage:**
    - User data saved locally in JSON files
    - Persistent across sessions
    - Each user has separate history
    """) 