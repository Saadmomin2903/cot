"""
Streamlit Frontend for Chain of Thought Pipeline.

Connects to the FastAPI backend to process text and visualize results.
Displays all 10 features in a clean, single-page layout.
"""

import os
import requests
import pandas as pd
import streamlit as st
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="CoT Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- Custom CSS for Cleaner UI ---
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stMetric {
        margin-bottom: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

def process_text(text: str, options: dict):
    """Call the backend API to process text."""
    try:
        payload = {
            "text": text,
            **options
        }
        response = requests.post(f"{API_URL}/process", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend API at {API_URL}. Is it running?")
        return None
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return None

def main():
    st.title("Rutwik")
    
    # --- Session State for Input ---
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    
    # --- Sidebar Options ---
    with st.sidebar:
        st.header("Pipeline Settings")
        
        with st.expander("Feature Toggles", expanded=True):
            enable_translate = st.checkbox("🌐 Translation", value=False)
            enable_semantic = st.checkbox("🧠 Semantic Clean", value=False)
            enable_ner = st.checkbox("🏷️ NER", value=True)
            enable_events = st.checkbox("📅 Events", value=True)
            enable_country = st.checkbox("🌏 Country ID", value=True)
            enable_sentiment = st.checkbox("💭 Sentiment", value=True)
            enable_summary = st.checkbox("📝 Summary", value=True)
            enable_relevancy = st.checkbox("🎯 Relevancy", value=True)
        
        st.divider()
        summary_style = st.selectbox("Summary Style", ["bullets", "paragraph", "executive", "headlines", "tldr"], index=0)
        custom_topics = st.text_input("Relevancy Topics", placeholder="Topics like Crypto, AI...")

    # --- Input Section ---
    text_input = st.text_area("Input Text", height=150, placeholder="Paste your text here...", key="user_input_area")
    process_btn = st.button("Analyze Text 🚀", type="primary", use_container_width=True)

    if process_btn and text_input:
        st.session_state.input_text = text_input # Save to session
        
        options = {
            "semantic_clean": enable_semantic,
            "ner": enable_ner,
            "relationships": True,
            "events": enable_events,
            "enable_country": enable_country,
            "sentiment": enable_sentiment,
            "summary": enable_summary,
            "summary_style": summary_style,
            "translate": enable_translate,
            "relevancy": enable_relevancy,
            "topics": custom_topics
        }
        
        with st.spinner("Processing..."):
            data = process_text(text_input, options)
            
        if data:
            results = data.get("results", {})
            duration = data.get("duration_ms", 0)
            
            # --- 1) Cleaned Text & Translation (First Row) ---
            st.subheader("📝 Cleaned Text & Pre-processing")
            
            # Find relevant keys
            clean_key = next((k for k in results.keys() if "text_cleaning" in k), None)
            semantic_key = next((k for k in results.keys() if "semantic_cleaning" in k), None)
            trans_key = next((k for k in results.keys() if "translation" in k), None)

            c1, c2 = st.columns(2)
            with c1:
                # Prefer semantic cleaned if available
                display_key = semantic_key if semantic_key and results.get(semantic_key, {}).get("status") == "success" else clean_key
                label = "Semantic" if display_key == semantic_key else "Basic"
                
                if display_key and results[display_key]["status"] == "success":
                    clean_out = results[display_key]["output"]
                    final_text = clean_out.get("cleaned_text", "")
                    st.markdown(f"**Cleaned Text** *({label})*")
                    st.code(final_text, language=None)
                    st.caption(f"Reduced by {clean_out.get('reduction_percent', 0):.1f}%")
                else:
                    st.warning("Text cleaning failed or not enabled.")
            
            with c2:
                if trans_key and results[trans_key]["status"] == "success":
                    trans_out = results[trans_key]["output"]
                    st.markdown("**Translated to English**")
                    if not trans_out.get("was_already_english"):
                         st.code(trans_out.get("translated_text", ""), language=None)
                         st.caption(f"From: {trans_out.get('source_language')}")
                    else:
                         st.info("Text was already English (No translation needed)")
                elif enable_translate:
                    st.warning("Translation failed (likely rate limit). Check logs.")

            st.divider()

            # --- 2) Key Metrics Dashboard (Middle Row) ---
            st.subheader("📊 Key Metrics")
            
            m1, m2, m3, m4, m5 = st.columns(5)
            
            # Country
            with m1:
                cnt_key = next((k for k in results.keys() if "country_id" in k), None)
                if cnt_key:
                    st.metric("Region", results[cnt_key]["output"].get("region", "N/A"))
                else:
                    st.metric("Region", "N/A")

            # Sentiment (Tone)
            with m2:
                sent_key = next((k for k in results.keys() if "sentiment" in k), None)
                if sent_key:
                    s_out = results[sent_key]["output"]
                    st.metric("Tone", s_out["sentiment"].title(), delta=s_out.get('emotion'))
                else:
                    st.metric("Tone", "N/A")

            # Language
            with m3:
                lang_key = next((k for k in results.keys() if "language" in k or "lang_" in k), None)
                if lang_key:
                    l_out = results[lang_key]["output"]
                    st.metric("Language", l_out["language_code"].upper(), l_out["script_type"])
                else:
                     st.metric("Language", "N/A")

            # Domain
            with m4:
                dom_key = next((k for k in results.keys() if "domain" in k), None)
                if dom_key:
                    dom_out = results[dom_key]["output"]
                    # Safe access for primary_domain
                    primary = dom_out.get("primary_domain")
                    if not primary:
                        # Try finding key with max score in all_domains
                        all_doms = dom_out.get("all_domains", {})
                        if all_doms:
                            primary = max(all_doms, key=all_doms.get)
                        else:
                            primary = "N/A"
                    
                    st.metric("Domain", str(primary).title())
                else:
                    st.metric("Domain", "N/A")

            # Latency
            with m5:
                st.metric("Latency", f"{duration}ms")

            # --- 3) Summary & Relevancy (Synthesis Row) ---
            st.divider()
            col_summ, col_chart = st.columns([1.5, 1])
            
            with col_summ:
                st.subheader("📝 Summary")
                sum_key = next((k for k in results.keys() if "summary" in k), None)
                if sum_key and results[sum_key]["status"] == "success":
                    sum_out = results[sum_key]["output"]
                    st.info(sum_out.get("summary", "No summary"))
                    # Show key points if available
                    if sum_out.get("key_points"):
                        with st.expander("Key Points"):
                            for p in sum_out["key_points"]:
                                st.markdown(f"- {p}")
                else:
                    st.warning("Summary not enabled.")

            with col_chart:
                st.subheader("🎯 Topic Relevancy")
                rel_key = next((k for k in results.keys() if "relevancy" in k), None)
                if rel_key and results[rel_key]["status"] == "success":
                    r_scores = results[rel_key]["output"].get("topic_scores", [])
                    if r_scores:
                        chart_data = {t['topic']: t['score'] for t in r_scores}
                        st.bar_chart(chart_data, color="#FF4B4B") # Streamlit red color
                        
                # Show Sentiment Aspects here if available (to fill space)
                sent_key = next((k for k in results.keys() if "sentiment" in k), None)
                if sent_key and results[sent_key]["status"] == "success":
                    aspects = results[sent_key]["output"].get("aspects", [])
                    if aspects:
                        with st.expander("❤️ Sentiment Aspects"):
                            st.dataframe(pd.DataFrame(aspects), use_container_width=True)

            # --- 4) Detailed Extraction (Bottom Row) ---
            st.divider()
            col_ner, col_evt = st.columns(2)
            
            with col_ner:
                st.subheader("🏷️ Entities & Relationships")
                ner_key = next((k for k in results.keys() if "ner" in k), None)
                if ner_key and results[ner_key]["status"] == "success":
                    ner_out = results[ner_key]["output"]
                    ents = ner_out.get("entities", [])
                    rels = ner_out.get("relationships", [])
                    
                    if ents:
                        st.caption(f"{len(ents)} Entities found")
                        df_ents = pd.DataFrame(ents)
                        # Fix column names if needed
                        cols = ["text", "type", "confidence"]
                        if "type" not in df_ents.columns and "label" in df_ents.columns:
                            df_ents["type"] = df_ents["label"]
                        
                        # Filter columns that exist
                        valid_cols = [c for c in cols if c in df_ents.columns]
                        st.dataframe(df_ents[valid_cols] if valid_cols else df_ents, use_container_width=True, height=200)
                    else:
                        st.text("No entities found.")
                        
                    if rels:
                         st.markdown("**Relationships**")
                         st.dataframe(pd.DataFrame(rels)[["source", "relation", "target"]], use_container_width=True, height=150)

            with col_evt:
                st.subheader("📅 Event Timeline")
                evt_key = next((k for k in results.keys() if "event" in k), None)
                if evt_key and results[evt_key]["status"] == "success":
                    evts = results[evt_key]["output"].get("events", [])
                    if evts:
                        # Simple timeline
                        st.dataframe(pd.DataFrame(evts)[["date", "description"]], use_container_width=True, height=350)
                    else:
                        st.text("No events detected.")
            
            # --- 5) Raw JSON Deep Dive ---
            with st.expander("🔍 View Raw Analysis JSON"):
                st.json(results)

if __name__ == "__main__":
    main()
