import streamlit as st
from ai_generated_analyzer import AIGeneratedAnalyzer

def main():
    # GUI
    st.set_page_config(page_title="AI Content Detector", page_icon="🔍", layout="wide")
    
    st.markdown("""
    <style>
        .main {padding: 2rem;}
        .score-box {padding: 1rem; border-radius: 10px; margin: 1rem 0;}
        .ai-generated {background-color: #ffe6e6; border: 2px solid #ff0000;}
        .human-written {background-color: #e6ffe6; border: 2px solid #00ff00;}
        .pattern-box {padding: 1rem; margin: 0.5rem 0; border-left: 4px solid #4a90e2;}
        [data-testid="stHorizontalBlock"] {gap: 2rem;}
    </style>
    """, unsafe_allow_html=True)

    st.title("🔍 AI Content Detection Tool")
    
    left_col, right_col = st.columns([2, 3])
    
    # left side => input 
    with left_col:
        st.subheader("Input Text")
        with st.form("analysis_form"):
            text_input = st.text_area("Paste your text below (min 200 characters):", 
                                     height=400,
                                     label_visibility="collapsed")
            analyze_button = st.form_submit_button("Analyze Content")

    # right side => output 
    with right_col:
        st.subheader("Analysis Results")
        
        if 'result' in st.session_state and st.session_state.result:
            result = st.session_state.result
            
            score_col, conclusion_col = st.columns([1, 2])
            with score_col:
                st.metric("AI Generation Score", f"{result['score']}/100")
                st.progress(result['score']/100)
            
            with conclusion_col:
                conclusion_class = "ai-generated" if "AI" in result['conclusion'] else "human-written"
                st.markdown(f"""
                <div class="score-box {conclusion_class}">
                    <h3>Conclusion:</h3>
                    <h2>{result['conclusion']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader("Detection Patterns")
            if result['patterns']:
                for pattern in result['patterns']:
                    confidence_color = {
                        "High": "🔴",
                        "Medium": "🟠",
                        "Low": "🟡"
                    }.get(pattern['confidence'], "⚪")
                    
                    with st.expander(f"{confidence_color} {pattern['pattern']} ({pattern['confidence']} Confidence)"):
                        st.markdown(f"""
                        <div class="pattern-box">
                            <p>{pattern['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No significant detection patterns found")

    if analyze_button:
        if len(text_input) < 200:
            st.error("Please input at least 200 characters for accurate analysis")
            st.session_state.result = None
            st.rerun()

        with st.spinner("Analyzing text with AWS Bedrock (Llama 3.1 405B)..."):
            try:
                analyzer = AIGeneratedAnalyzer()
                result = analyzer.analyze_text(text_input)
                st.session_state.result = result
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.session_state.result = None
                st.rerun()

if __name__ == "__main__":
    main()