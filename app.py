import streamlit as st
import openai
from pathlib import Path
import PyPDF2
import io
import time
from typing import List, Dict
from dotenv import load_dotenv
import os
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import nltk

# Download required NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

# Load Spacy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="AdaptiveLOM",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stTextInput > label, .stTextArea > label {
        font-size: 1.2rem;
        color: #0f52ba;
        font-weight: 600;
    }
    .stButton > button {
        background-color: #0f52ba;
        color: white;
        padding: 0.5rem 2rem;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class CVProcessor:
    def __init__(self):
        download_nltk_resources()
        self.nlp = load_spacy_model()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return filtered_tokens

    def extract_named_entities(self, text):
        doc = self.nlp(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'DATE': [],
            'WORK_EXPERIENCE': [],
            'DEGREES': []
        }
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        return entities

    def extract_key_phrases(self, text):
        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out()

    def process_cv(self, text):
        preprocessed_text = self.preprocess_text(text)
        named_entities = self.extract_named_entities(text)
        key_phrases = self.extract_key_phrases(" ".join(preprocessed_text))
        sentences = sent_tokenize(text)
        summary = " ".join(sentences[:5])  # Basic summarization
        return {
            'Named Entities': named_entities,
            'Key Phrases': key_phrases,
            'Summary': summary
        }

class LoMGenerator:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.cv_processor = CVProcessor()
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def generate_section(self, prompt: str, cv_text: str, previous_sections: str = "", 
                        university: str = "", program: str = "", specialization: str = "") -> str:
        """Generate a section of the LoM using GPT-4"""
        # Process CV text
        cv_analysis = self.cv_processor.process_cv(cv_text)
        
        context = f"""
        CV Summary: {cv_analysis['Summary']}
        Key Phrases: {', '.join(cv_analysis['Key Phrases'])}
        Named Entities: {cv_analysis['Named Entities']}
        
        University: {university}
        Program: {program}
        Specialization: {specialization}
        Previous Sections: {previous_sections}
        
        Full CV Content: {cv_text}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in writing Letters of Motivation."},
                {"role": "user", "content": f"{prompt}\n\nContext:\n{context}"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content

    def evaluate_and_improve(self, complete_lom: str) -> tuple[str, str]:
        """Evaluate the complete LoM and generate an improved version"""
        evaluation_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating Letters of Motivation."},
                {"role": "user", "content": f"Evaluate this Letter of Motivation and list any issues or areas for improvement:\n\n{complete_lom}"}
            ],
            temperature=0.7
        )
        
        issues = evaluation_response.choices[0].message.content
        
        improvement_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in improving Letters of Motivation."},
                {"role": "user", "content": f"Improve this Letter of Motivation based on these issues:\n\nOriginal Letter:\n{complete_lom}\n\nIssues to Address:\n{issues}"}
            ],
            temperature=0.7
        )
        
        return issues, improvement_response.choices[0].message.content

def main():
    st.title("📝 AdaptiveLOM")
    st.markdown("### Transform your CV into a compelling Letter of Motivation")
    
    # API Key handling
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if not api_key:
            st.warning("Please enter your OpenAI API key to proceed.")
            st.info("You can get your API key from: https://platform.openai.com/api-keys")
            return
    
    # Initialize session state with API key
    if 'lom_generator' not in st.session_state:
        st.session_state.lom_generator = LoMGenerator(api_key)
    
    # Create columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        university = st.text_input("University Name*", placeholder="e.g., Technical University of Munich")
        program = st.text_input("Program Name*", placeholder="e.g., Computer Science")
    
    with col2:
        specialization = st.text_input("Specialization (Optional)", placeholder="e.g., Machine Learning")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CV (PDF format)", type=['pdf'])
    
    if uploaded_file and university and program:
        cv_text = st.session_state.lom_generator.extract_text_from_pdf(uploaded_file.read())
        
        # Process and display CV analysis
        cv_analysis = st.session_state.lom_generator.cv_processor.process_cv(cv_text)
        
        with st.expander("📊 CV Analysis Results"):
            st.markdown("### Key Phrases")
            st.write(", ".join(cv_analysis['Key Phrases']))
            
            st.markdown("### Named Entities")
            for entity_type, entities in cv_analysis['Named Entities'].items():
                if entities:
                    st.markdown(f"**{entity_type}:** {', '.join(set(entities))}")
            
            st.markdown("### CV Summary")
            st.write(cv_analysis['Summary'])
        
        if st.button("Generate Letter of Motivation"):
            with st.spinner("Generating your Letter of Motivation..."):
                # Generate sections sequentially
                sections = {}
                
                # 1. Introduction and Academic Background
                st.markdown("### 🎓 Generating Introduction and Academic Background...")
                sections['intro'] = st.session_state.lom_generator.generate_section(
                    "Write an engaging introduction and academic background section for a Letter of Motivation.",
                    cv_text, "", university, program, specialization
                )
                
                # 2. Research and Work Experience
                st.markdown("### 💼 Generating Research and Work Experience...")
                sections['experience'] = st.session_state.lom_generator.generate_section(
                    "Write about the research and work experience relevant to this application.",
                    cv_text, sections['intro'], university, program, specialization
                )
                
                # 3. Professional Goals
                st.markdown("### 🎯 Generating Professional Goals...")
                sections['goals'] = st.session_state.lom_generator.generate_section(
                    "Write about short-term and long-term professional goals.",
                    cv_text, sections['intro'] + "\n" + sections['experience'],
                    university, program, specialization
                )
                
                # 4. Why This Program
                st.markdown("### 🌟 Generating Program Fit...")
                sections['why_program'] = st.session_state.lom_generator.generate_section(
                    "Explain why this specific program and university is the perfect fit.",
                    cv_text, "\n".join(sections.values()),
                    university, program, specialization
                )
                
                # Combine all sections
                complete_lom = "\n\n".join(sections.values())
                
                # Evaluate and improve
                st.markdown("### 📊 Evaluating and Improving...")
                issues, improved_lom = st.session_state.lom_generator.evaluate_and_improve(complete_lom)
                
                # Display results
                st.markdown("## 📝 Your Letter of Motivation")
                
                # Create tabs for different versions
                tab1, tab2, tab3 = st.tabs(["Final Version", "Initial Draft", "Evaluation"])
                
                with tab1:
                    st.markdown("### Final Improved Version")
                    st.markdown(improved_lom)
                    st.download_button(
                        label="Download Final LoM",
                        data=improved_lom,
                        file_name="Letter_of_Motivation_Final.txt",
                        mime="text/plain"
                    )
                
                with tab2:
                    st.markdown("### Initial Draft")
                    st.markdown(complete_lom)
                
                with tab3:
                    st.markdown("### Evaluation and Issues")
                    st.markdown(issues)

if __name__ == "__main__":
    main()