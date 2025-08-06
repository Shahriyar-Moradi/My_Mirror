"""
Streamlit UI for Conversation Token Extraction using langextract.

This is a simplified version that focuses on single-turn extraction and visualization.

Run with: streamlit run streamlit_app_langextract.py
"""

import streamlit as st
import textwrap
import os
import uuid
from dotenv import load_dotenv

# Import langextract
try:
    import langextract as lx
except ImportError:
    st.error("The `langextract` library is not installed. Please install it by running: `pip install langextract`")
    st.stop()

# Initialize
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "")

# Page Configuration
st.set_page_config(
    page_title="Entity and Emotion Analyzer",
    page_icon="✨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .langextract-badge {
        background: linear-gradient(90deg, #84fab0, #8fd3f4);
        color: #333;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Status
    if api_key:
        st.success("✅ API Key loaded from .env file")
    else:
        st.error("❌ No API Key found. Please set GEMINI_API_KEY in your .env file")
    
    # Model Selection
    model_id = st.selectbox(
        "Select Model:",
        ("gemini-1.5-pro-latest", "gemini-2.5-flash", "gemini-2.5-pro"),
        index=1,
        help="Select the Gemini model for extraction."
    )

# Define extraction prompt
prompt = textwrap.dedent("""
Extract activities, people, characters, and emotions in order of appearance.
Use exact text for extractions. Do not paraphrase or overlap entities.
Provide meaningful attributes for each entity to add context.""")

# Define example for few-shot learning
examples = [
    lx.data.ExampleData(
        text=(
            "As Sarah crossed the marathon finish line, I felt a wave of pride, like a "
            "superhero watching their sidekick succeed. We celebrated with a huge "
            "pizza party afterwards."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="person",
                extraction_text="Sarah",
                attributes={"relationship": "unspecified"},
            ),
            lx.data.Extraction(
                extraction_class="activity",
                extraction_text="crossed the marathon finish line",
                attributes={"type": "achievement", "sport": "running"},
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="pride",
                attributes={"intensity": "high"},
            ),
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="superhero",
                attributes={"archetype": "mentor/protector"},
            ),
            lx.data.Extraction(
                extraction_class="activity",
                extraction_text="celebrated with a huge pizza party",
                attributes={"type": "social gathering"},
            ),
        ],
    )
]

# Main UI
st.title("✨ Entity and Emotion Extraction with LangExtract")
st.markdown('<div class="langextract-badge">Powered by LangExtract & Gemini</div>', unsafe_allow_html=True)

# Text Input
input_text = st.text_area(
    "Enter the text you want to analyze:",
    value=(
        "I felt a huge sense of pride watching my brother, Alex, graduate. We all went out for a celebratory dinner after."
    ),
    height=150,
)

# Process Button
if st.button("Extract & Visualize", type="primary", use_container_width=True):
    if not api_key:
        st.error("Please set GEMINI_API_KEY in your .env file to continue.")
    elif not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner(f"Running extraction with `{model_id}`..."):
            try:
                # Run extraction
                result = lx.extract(
                    text_or_documents=input_text,
                    prompt_description=prompt,
                    api_key=api_key,
                    examples=examples,
                    model_id=model_id,
                )

                # Save results temporarily for visualization
                temp_filename = f"temp_results_{uuid.uuid4()}.jsonl"
                lx.io.save_annotated_documents([result], output_name=temp_filename, output_dir=".")

                # Display visualization
                if os.path.exists(temp_filename):
                    st.header("🔎 Visualization")
                    html_content = lx.visualize(temp_filename)
                    st.components.v1.html(html_content.data, height=400, scrolling=True)
                    
                    # Clean up the temporary file
                    os.remove(temp_filename)
                else:
                    st.error("Could not save temporary file for visualization.")

                # Display extraction details
                st.header("Extraction Details")
                if result.extractions:
                    for extraction in result.extractions:
                        st.subheader(f"**{extraction.extraction_class.capitalize()}**: `{extraction.extraction_text}`")
                        if extraction.attributes:
                            for key, value in extraction.attributes.items():
                                st.markdown(f"- **{key.replace('_', ' ').capitalize()}**: {value}")
                        else:
                            st.markdown("No attributes found.")
                else:
                    st.info("No entities were extracted from the text.")

                # Show raw data
                with st.expander("View Raw Extraction Data"):
                    extraction_data = [{
                        "class": extraction.extraction_class,
                        "text": extraction.extraction_text,
                        "attributes": extraction.attributes or {}
                    } for extraction in result.extractions]
                    
                    st.json({
                        "text": result.text,
                        "extractions": extraction_data
                    })

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
