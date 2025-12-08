import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import praw

import google.generativeai as genai

# REQUIRED FOR ROBUST API CALLS AND ERROR HANDLING
from google.api_core import exceptions as google_exceptions
import time 

GEMINI_API_KEY = "AIzaSyDqLKOAB0bExqn_FN4hhHkBOcefpHa2huI"

# Reddit credentials
REDDIT_CLIENT_ID = "iQ1lCggwRm0-cfdbHi1yMQ"
REDDIT_CLIENT_SECRET = "tonP3SmxXYK_ud-aE1mctHjOS1kphw"
REDDIT_USER_AGENT = "script:scraper:v1.0 (by u/MeetSpare519)"

# ROBUST GEMINI API SETUP & FUNCTIONS (UPDATED)

import os
from google.api_core import exceptions as google_exceptions

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY if 'GEMINI_API_KEY' in globals() else None)

# Try to configure the genai client, but do not crash the app if it fails.
GENAI_AVAILABLE = False
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GENAI_AVAILABLE = True
    else:
        # still allow the app to run — AI features will return helpful messages
        GENAI_AVAILABLE = False
except Exception as e:
    GENAI_AVAILABLE = False
    st.warning(f"Gemini configuration failed: {e}. AI features will be disabled until you set a valid key.")

def get_valid_model_name(requested_model: str) -> str:
    """
    Map common/old model aliases to currently recommended models.
    If you need a different model, call genai.list_models() (see message below).
    """
    model_map = {
        # map old/retired aliases to recommended replacements
        "gemini-1.5-flash": "gemini-2.5-flash-lite",
        "gemini-1.5-pro": "gemini-2.5-pro",
        "gemini-1.5-flash-latest": "gemini-2.5-flash-lite",
        "gemini-1.5-pro-latest": "gemini-2.5-pro",
        # keep new names as-is
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
        "gemini-2.5-pro": "gemini-2.5-pro",
    }
    return model_map.get(requested_model, requested_model)

def list_available_models():
    """
    Helper that attempts to call the SDK's model listing. Returns a short string for UI.
    """
    if not GENAI_AVAILABLE:
        return "Gemini not configured (no API key). Set GEMINI_API_KEY environment variable."
    try:
        models = genai.list_models()  
        # build a short readable list (first 40 models)
        names = [m.name for m in models][:40]
        return "Available models: " + ", ".join(names)
    except Exception as e:
        return f"Failed to list models: {type(e).__name__}: {e}"

def generate_with_retry(model_name: str, prompt: str, retries: int = 3):
    """
    Robust generator wrapper. Returns text on success or a helpful error message on failure.
    """
    if not GENAI_AVAILABLE:
        return ("Gemini API not configured. Set the GEMINI_API_KEY environment variable "
                "and ensure the key has access to desired Gemini models (or run genai.list_models()).")

    valid_model = get_valid_model_name(model_name)

    # Defensive: ensure prompt isn't absurdly long for the wrapper itself
    if isinstance(prompt, (list, tuple)):
        # join lists if passed accidentally
        prompt = "\n".join(str(p) for p in prompt)
    if len(prompt) > 12000:
        prompt = prompt[:12000]

    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            
            model = genai.GenerativeModel(valid_model)

            try:
                response = model.generate_content(contents=prompt)
            except TypeError:
                # fallback to older positional arg style
                response = model.generate_content(prompt)
            # the response object may have .text or .output depending on SDK version
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            # some SDK responses return 'outputs' or 'candidates'
            if hasattr(response, "outputs") and response.outputs:
                # try to extract textual output
                out = []
                for o in response.outputs:
                    if hasattr(o, "content"):
                        # content
                        out.append(o.content)
                    elif isinstance(o, str):
                        out.append(o)
        except Exception as e:
            last_exception = e
            time.sleep(1)
            continue
# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Session state initialization
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = ""
if "keyword_summary" not in st.session_state:
    st.session_state.keyword_summary = ""
if "extracted_keywords" not in st.session_state:
    st.session_state.extracted_keywords = "N/A"

# Page config
st.set_page_config(page_title="Reddit Sentiment Dashboard", layout="wide")

# Sidebar for Data Ingestion and Configuration
st.sidebar.title("Data Sources")

# --- Reddit Scraper ---
with st.sidebar.expander("🌐 Scrape Reddit", expanded=True):
    subreddit_name = st.text_input("Subreddit Name", value="python")
    post_limit = st.slider("Number of Posts", 1, 100, 10)
    if st.button("Scrape Subreddit"):
        try:
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
            )
            subreddit = reddit.subreddit(subreddit_name)
            comments = []
            with st.spinner(f"Scraping r/{subreddit_name}..."):
                for post in subreddit.hot(limit=post_limit):
                    if post.title:
                        comments.append(post.title)
                    if post.selftext:
                        comments.append(post.selftext)
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list():
                        comments.append(comment.body)

            st.session_state.df = pd.DataFrame(comments, columns=["comment"])
            st.success(f"Scraped {len(comments)} comments from r/{subreddit_name}")
            st.session_state.ai_summary = ""
            st.session_state.keyword_summary = ""
            st.session_state.extracted_keywords = "N/A"
        except Exception as e:
            st.error(f"Error: {e}")

# --- File Upload ---
with st.sidebar.expander("📂 Upload a File"):
    uploaded = st.file_uploader("Upload CSV, TXT, Excel, or JSON", type=["csv", "txt", "xls", "xlsx", "json"])
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            elif uploaded.name.endswith(".json"):
                df = pd.read_json(uploaded, lines=True)
            elif uploaded.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(uploaded)
            else: 
                lines = uploaded.read().decode("utf-8").splitlines()
                df = pd.DataFrame(lines, columns=["comment"])

            if "comment" not in df.columns:
                for col in df.columns:
                    if df[col].dtype == "object": 
                        df.rename(columns={col: "comment"}, inplace=True)
                        break
            if "comment" not in df.columns:
                raise ValueError("No text column found. Please name one 'comment'.")
            
            st.session_state.df = df
            st.success("File uploaded and processed.")
            st.session_state.ai_summary = ""
            st.session_state.keyword_summary = ""
            st.session_state.extracted_keywords = "N/A"
        except Exception as e:
            st.error(f"Error: {e}")

# --- Manual Entry ---
with st.sidebar.expander("✍️ Manual Entry"):
    manual_comment = st.text_area("Enter a comment")
    if st.button("Add Comment"):
        if manual_comment.strip():
            new = pd.DataFrame({"comment": [manual_comment.strip()]})
            st.session_state.df = pd.concat([st.session_state.df, new], ignore_index=True)
            st.success("Comment added.")

# --- Clear Data ---
if st.sidebar.button("🧹 Clear All Data"):
    st.session_state.df = pd.DataFrame()
    st.session_state.ai_summary = ""
    st.session_state.keyword_summary = ""
    st.session_state.extracted_keywords = "N/A"
    st.sidebar.success("Cleared.")

# --- Gemini AI Toggle ---
st.sidebar.header("💎 Gemini AI Insights")

use_ai = st.sidebar.checkbox("Enable Gemini Insights", value=False)
# Use stable model names
ai_model_choice = st.sidebar.selectbox(
    "Choose Gemini Model",
    ["gemini-1.5-flash", "gemini-1.5-pro"],
    index=0
)
st.session_state.ai_model_choice = ai_model_choice # Store the stable choice

if use_ai:
    st.sidebar.warning(
        "⚠️ 'Pro' is smarter but slower. 'Flash' is faster."
    )

    # --- Keyword Extraction ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Extract Keywords & Entities")
    
    def get_keywords_and_entities(df, model_choice):
        if df.empty:
            return "N/A"
        
        sample_comments = " ".join(df["comment"].dropna().astype(str).sample(min(50, len(df))).tolist())
        prompt = f"""
        Identify the top 10 most important keywords and named entities from the text below.
        List them as a comma-separated string. Entities in brackets.
        Text: {sample_comments}
        """
        # Calls the robust generator function
        return generate_with_retry(model_choice, prompt)
    
    if st.sidebar.button("Extract Keywords"):
        with st.spinner("Extracting..."):
            st.session_state.extracted_keywords = get_keywords_and_entities(st.session_state.df, st.session_state.ai_model_choice)
    
# Main Dashboard
st.title("📊 Reddit Sentiment Analysis Dashboard")

if not st.session_state.df.empty and "comment" in st.session_state.df.columns:
    df = st.session_state.df.copy()
    df.dropna(subset=["comment"], inplace=True)
    df.drop_duplicates(subset=["comment"], inplace=True)

    def get_sentiment(text):
        score = analyzer.polarity_scores(str(text))
        compound = score["compound"]
        sentiment = "Neutral"
        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        return sentiment, compound

    df[["sentiment", "score"]] = df["comment"].apply(lambda x: pd.Series(get_sentiment(x)))

    # --- Sentiment Filters ---
    st.sidebar.header("Filter by Sentiment")
    show_positive = st.sidebar.checkbox("Positive", True)
    show_neutral = st.sidebar.checkbox("Neutral", True)
    show_negative = st.sidebar.checkbox("Negative", True)

    sentiments = []
    if show_positive: sentiments.append("Positive")
    if show_neutral: sentiments.append("Neutral")
    if show_negative: sentiments.append("Negative")

    df_filtered = df[df["sentiment"].isin(sentiments)]

    # --- Keyword Search ---
    search_term = st.text_input("🔍 Search for a keyword to get specific AI insights:", "")
    
    # Summary Cards
    col1, col2, col3 = st.columns(3)
    total = len(df_filtered)
    pos_pct = df_filtered["sentiment"].eq("Positive").mean() * 100 if total > 0 else 0
    neg_pct = df_filtered["sentiment"].eq("Negative").mean() * 100 if total > 0 else 0

    with col1:
        st.metric("Total Comments", total)
    with col2:
        st.metric("Positive", f"{pos_pct:.1f}%")
    with col3:
        st.metric("Negative", f"{neg_pct:.1f}%")

    # Data Table
    st.subheader("💬 Filtered Comments & Sentiment")
    st.dataframe(df_filtered[["comment", "sentiment", "score"]], use_container_width=True)

    # Plot
    st.subheader("📊 Sentiment Distribution")
    # Removed the trailing comma in the px.histogram call to resolve SyntaxError
    fig = px.histogram(
        df_filtered,
        x="sentiment",
        color="sentiment",
        category_orders={"sentiment": ["Positive", "Neutral", "Negative"]},
        color_discrete_map={"Positive": "#2EBE83", "Neutral": "#FFC107", "Negative": "#E94B4B"},
        title="Sentiment Count" 
    ) 
    st.plotly_chart(fig, use_container_width=True)

    # Word Cloud
    st.subheader("☁️ Word Cloud")
    if not df_filtered.empty:
        text = " ".join(df_filtered["comment"].astype(str))
        wc = WordCloud(width=800, height=400, background_color="black", colormap="plasma").generate(text)
        st.image(wc.to_array(), use_column_width=True)

    # --- AI Insights ---
    if use_ai:
        def get_ai_summary(df, search_term, model_choice):
            if df.empty:
                return "No data to analyze."

            if search_term:
                df = df[df["comment"].str.contains(search_term, case=False, na=False)]
                if df.empty:
                    return f"No comments found containing the keyword '{search_term}'."
            
            sample_comments = df["comment"].dropna().astype(str).sample(min(30, len(df))).tolist()
            comments_text = "\n".join([f"- {c}" for c in sample_comments])
            
            if search_term:
                prompt = f"""
                You are an expert sentiment analyst.
                Based on these Reddit comments, provide a detailed sentiment analysis specifically focused on the keyword "{search_term}".
                Your analysis should include: 1. A summary of the sentiment related to the keyword. 2. Key topics/themes. 3. Positive, negative, and neutral examples. 4. Any interesting patterns.
                Sample comments: {comments_text}
                """
            else:
                prompt = f"""
                You are an expert sentiment analyst.
                Based on these Reddit comments, provide a detailed sentiment analysis of the dataset as a whole.
                Your analysis should include: 1. A summary of the overall sentiment trends. 2. Key topics/themes. 3. Positive, negative, and neutral examples. 4. Any interesting patterns.
                Sample comments: {comments_text}
                """

            # Calls the robust generator function
            return generate_with_retry(model_choice, prompt)

        st.subheader("🤖 AI Insights on Sentiment")

        # Display either general or keyword-specific analysis
        if search_term:
            if st.button(f"Analyze '{search_term}'"):
                 with st.spinner(f"Generating insights for '{search_term}' with {st.session_state.ai_model_choice}..."):
                    st.session_state.keyword_summary = get_ai_summary(df_filtered, search_term, st.session_state.ai_model_choice)
            
            # Display result or error
            if st.session_state.keyword_summary and not st.session_state.keyword_summary.startswith("Error"):
                 st.markdown(st.session_state.keyword_summary)
            elif st.session_state.keyword_summary.startswith("Error"):
                 st.error(st.session_state.keyword_summary)

        else:
            if st.button("Generate General Summary"):
                with st.spinner(f"Generating general insights with {st.session_state.ai_model_choice}..."):
                    st.session_state.ai_summary = get_ai_summary(df_filtered, None, st.session_state.ai_model_choice)
            
            # Display result or error
            if st.session_state.ai_summary and not st.session_state.ai_summary.startswith("Error"):
                st.markdown(st.session_state.ai_summary)
            elif st.session_state.ai_summary.startswith("Error"):
                 st.error(st.session_state.ai_summary)


        # --- Display the extracted keywords ---
        st.subheader("🔑 Key Entities and Keywords")
        st.markdown(st.session_state.extracted_keywords)
        
    # --- Export ---
    st.subheader("⬇ Export Results")
    col1, col2 = st.columns(2)

    export_df = df_filtered.copy()
    export_df["ai_insights"] = st.session_state.keyword_summary if search_term else st.session_state.ai_summary
    csv = export_df.to_csv(index=False).encode("utf-8")
    col1.download_button("Download CSV", csv, "sentiment_results.csv", "text/csv")

    # --- Enhanced PDF Report ---
    def create_pdf_report(df, ai_summary, extracted_keywords):
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        elements.append(Paragraph("Sentiment Analysis Report", styles['Heading1']))
        elements.append(Paragraph("A comprehensive analysis of public sentiment.", styles['Normal']))
        elements.append(Spacer(1, 24))

        # Table of Contents
        elements.append(Paragraph("<b>Table of Contents</b>", styles['Heading2']))
        toc_style = styles['Normal']
        toc_style.leftIndent = 20
        elements.append(Paragraph("- Overall Sentiment Summary", toc_style))
        elements.append(Paragraph("- Detailed Comments and Sentiment Scores", toc_style))
        elements.append(Paragraph("- AI-Powered Insights", toc_style))
        elements.append(Paragraph("- Key Entities and Keywords", toc_style))
        elements.append(Spacer(1, 12))
        
        # Overall Summary
        elements.append(Paragraph("<b>1. Overall Sentiment Summary</b>", styles['Heading2']))
        elements.append(Spacer(1, 12))
        data = [['Total Comments', 'Positive %', 'Negative %'],
                [len(df), f"{df['sentiment'].eq('Positive').mean() * 100:.1f}%", f"{df['sentiment'].eq('Negative').mean() * 100:.1f}%"]]
        table_style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                  ('GRID', (0, 0), (-1, -1), 1, colors.black),
                                  ('BOX', (0, 0), (-1, -1), 1, colors.black),
                                  ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                  ('ALIGN', (0, 0), (-1, -1), 'CENTER')])
        table = Table(data, style=table_style)
        elements.append(table)
        elements.append(Spacer(1, 24))

        # Detailed Comments (Limited to prevent massive PDF file size)
        elements.append(Paragraph("<b>2. Detailed Comments and Sentiment Scores (Preview first 50)</b>", styles['Heading2']))
        elements.append(Spacer(1, 12))
        for _, row in df.head(50).iterrows(): 
            clean_cmt = str(row['comment']).replace("<", "&lt;").replace(">", "&gt;")[:300]
            elements.append(Paragraph(f"<b>Comment:</b> {clean_cmt}...", styles['Normal']))
            elements.append(Paragraph(f"<b>Sentiment:</b> {row['sentiment']} (Score: {row['score']:.2f})", styles['Normal']))
            elements.append(Spacer(1, 12))
        elements.append(Spacer(1, 24))

        # AI-Powered Insights
        elements.append(Paragraph("<b>3. AI-Powered Insights</b>", styles['Heading2']))
        elements.append(Spacer(1, 12))
        # Basic sanitization for PDF
        clean_ai = str(ai_summary).replace("#", "").replace("*", "") 
        elements.append(Paragraph(clean_ai, styles['Normal']))
        elements.append(Spacer(1, 24))

        # Key Entities and Keywords
        elements.append(Paragraph("<b>4. Key Entities and Keywords</b>", styles['Heading2']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"<b>Keywords:</b> {extracted_keywords}", styles['Normal']))
        elements.append(Spacer(1, 24))
        
        doc.build(elements)
        return pdf_buffer.getvalue()

    pdf_output = create_pdf_report(df_filtered, 
                                   st.session_state.keyword_summary if search_term else st.session_state.ai_summary,
                                   st.session_state.extracted_keywords)
    col2.download_button("Download PDF", pdf_output, "sentiment_report.pdf", "application/pdf")

else:
    st.info("👋 Start by scraping a subreddit, uploading a file, or entering a comment manually.")