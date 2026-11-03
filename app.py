import os
import io
import re
import json
import hashlib
from datetime import datetime
from xml.sax.saxutils import escape

import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #101827, #17243a);
        color: white;
    }

    h1, h2, h3, p, label {
        color: white !important;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: #69b7ff;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .feature-card {
        background: #24344d;
        border: 1px solid #385475;
        padding: 22px;
        border-radius: 15px;
        min-height: 130px;
    }

    .answer-card {
        background: #1d3048;
        border-left: 4px solid #52a8ff;
        padding: 18px;
        border-radius: 10px;
        margin-top: 12px;
    }

    [data-testid="stMetric"] {
        background: #24344d;
        padding: 15px;
        border-radius: 12px;
    }

    .stButton button {
        border-radius: 9px;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        background: #111d30;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "chunks": [],
    "vectorizer": None,
    "matrix": None,
    "doc_name": None,
    "doc_hash": None,
    "chat_history": [],
    "summary": "",
    "quiz": "",
    "report": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# GROQ API KEY
# ============================================================

def get_api_key():
    # First check Streamlit Cloud secrets
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    # Then check environment variables
    return os.environ.get("GROQ_API_KEY", "")


# ============================================================
# GROQ AI FUNCTION
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"

def ask_groq(prompt, system_prompt=None, api_key=None):

    key = api_key or get_api_key()

    if not key:
        raise ValueError(
            "Groq API key is missing. Enter your API key in the sidebar."
        )

    client = Groq(api_key=key)

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=1800
    )

    return response.choices[0].message.content


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf(uploaded_file):

    pdf = PdfReader(io.BytesIO(uploaded_file.getvalue()))

    pages = []

    for page_number, page in enumerate(pdf.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    if not pages:
        raise ValueError(
            "No readable text found. Please upload a text-based PDF."
        )

    return pages


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(pages, chunk_size=1800, overlap=250):

    chunks = []

    for page in pages:

        text = re.sub(r"\s+", " ", page["text"]).strip()

        start = 0
        local_chunk = 1

        while start < len(text):

            end = start + chunk_size
            content = text[start:end]

            if len(content.strip()) > 30:

                chunks.append({
                    "text": content,
                    "page": page["page"],
                    "chunk": local_chunk
                })

                local_chunk += 1

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# ============================================================
# TF-IDF DOCUMENT RETRIEVAL
# ============================================================

def build_search_index(chunks):

    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


def retrieve_context(question, top_k=5):

    if not st.session_state.chunks:
        return []

    vectorizer = st.session_state.vectorizer
    matrix = st.session_state.matrix

    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    best_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in best_indices:

        if scores[index] > 0:

            chunk = st.session_state.chunks[index]

            results.append({
                "text": chunk["text"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "score": float(scores[index])
            })

    return results


# ============================================================
# DOCUMENT CONTEXT
# ============================================================

def get_full_document_text():

    return "\n\n".join(
        f"[Page {page['page']}]\n{page['text']}"
        for page in st.session_state.get("pages", [])
    )


# ============================================================
# PDF REPORT CREATION
# ============================================================

def create_pdf_report(title, content):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=55,
        leftMargin=55,
        topMargin=55,
        bottomMargin=55
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        alignment=TA_CENTER,
        textColor="#174A78",
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=18,
        textColor="#174A78",
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=8
    )

    story = []

    story.append(Paragraph(escape(title), title_style))

    story.append(
        Paragraph(
            "Generated by DocuMind AI",
            ParagraphStyle(
                "small",
                parent=styles["Normal"],
                alignment=TA_CENTER,
                textColor="#666666"
            )
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Date: " + datetime.now().strftime("%d-%m-%Y"),
            body_style
        )
    )

    story.append(Spacer(1, 10))

    for line in content.splitlines():

        line = line.strip()

        if not line:
            story.append(Spacer(1, 5))
            continue

        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            story.append(
                Paragraph(escape(heading), heading_style)
            )

        elif line.startswith("- ") or line.startswith("* "):
            item = escape(line[2:])
            story.append(
                Paragraph("&#8226; " + item, body_style)
            )

        else:
            clean_line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
            clean_line = re.sub(r"`(.*?)`", r"\1", clean_line)

            story.append(
                Paragraph(escape(clean_line), body_style)
            )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 DocuMind AI")

    st.caption("Your intelligent document assistant")

    st.divider()

    api_key = get_api_key()

    if not api_key:

        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="Enter your Groq API key"
        )

        st.caption(
            "Your key is used for this session. "
            "Do not share it publicly."
        )

    else:
        st.success("Groq API key configured")

    st.divider()

    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a text-based PDF document."
    )

    if uploaded_file:

        file_bytes = uploaded_file.getvalue()

        if len(file_bytes) > 20 * 1024 * 1024:
            st.error("Maximum PDF size is 20 MB.")

        else:

            current_hash = hashlib.md5(file_bytes).hexdigest()

            if current_hash != st.session_state.doc_hash:

                with st.spinner("Reading and processing your PDF..."):

                    try:

                        pages = extract_pdf(uploaded_file)

                        chunks = create_chunks(pages)

                        if not chunks:
                            raise ValueError(
                                "The PDF does not contain enough readable text."
                            )

                        vectorizer, matrix = build_search_index(chunks)

                        st.session_state.pages = pages
                        st.session_state.chunks = chunks
                        st.session_state.vectorizer = vectorizer
                        st.session_state.matrix = matrix
                        st.session_state.doc_name = uploaded_file.name
                        st.session_state.doc_hash = current_hash

                        st.session_state.chat_history = []
                        st.session_state.summary = ""
                        st.session_state.quiz = ""
                        st.session_state.report = ""

                        st.success("PDF processed successfully!")

                    except Exception as e:
                        st.error(f"PDF processing failed: {e}")

    if st.session_state.doc_name:

        st.divider()

        st.success("Document ready")

        st.write("📄", st.session_state.doc_name)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Pages",
                len(st.session_state.pages)
            )

        with col2:
            st.metric(
                "Chunks",
                len(st.session_state.chunks)
            )

        if st.button("🗑️ Remove document", use_container_width=True):

            for key, value in defaults.items():
                st.session_state[key] = value

            st.session_state.pop("pages", None)

            st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 DocuMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Understand documents. Ask questions. Learn smarter.'
    '</div>',
    unsafe_allow_html=True
)

if not st.session_state.doc_name:

    st.info(
        "👈 Upload a PDF from the sidebar to start using DocuMind AI."
    )

    st.markdown("### ✨ What can DocuMind AI do?")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            '<div class="feature-card">'
            '<h3>💬 PDF Chat</h3>'
            '<p>Ask questions and get document-based answers.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            '<div class="feature-card">'
            '<h3>📝 Summary</h3>'
            '<p>Convert long documents into clear summaries.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            '<div class="feature-card">'
            '<h3>🧩 Quiz</h3>'
            '<p>Generate MCQs to test your understanding.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            '<div class="feature-card">'
            '<h3>📑 Reports</h3>'
            '<p>Create downloadable PDF reports.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    st.stop()


# ============================================================
# MAIN NAVIGATION
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 PDF Chat",
    "📝 AI Summary",
    "🧩 Quiz Generator",
    "📑 PDF Report"
])


# ============================================================
# FEATURE 1: PDF CHAT
# ============================================================

with tab1:

    st.subheader("💬 Chat with your document")

    st.write(
        "Ask anything about the uploaded PDF. "
        "DocuMind AI retrieves relevant sections before answering."
    )

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:

                with st.spinner("Searching your document and generating an answer..."):

                    results = retrieve_context(question)

                    if not results:

                        answer = (
                            "I could not find sufficiently relevant "
                            "information in your uploaded document. "
                            "Please try a more specific question."
                        )

                    else:

                        context = "\n\n".join(
                            f"[Page {r['page']}, Chunk {r['chunk']}]\n"
                            f"{r['text']}"
                            for r in results
                        )

                        answer = ask_groq(
                            prompt=f"""
Use ONLY the following retrieved document context
to answer the user's question.

If the answer is not supported by the context,
clearly say that the document does not provide
enough information.

Cite page numbers in your answer using [Page X].

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear, beginner-friendly answer.
""",
                            system_prompt=(
                                "You are DocuMind AI, a helpful document "
                                "assistant. Do not invent document facts."
                            ),
                            api_key=api_key
                        )

                        sources = sorted(
                            set(r["page"] for r in results)
                        )

                        answer += "\n\n**Retrieved source pages:** " + ", ".join(
                            str(p) for p in sources
                        )

                st.markdown(answer)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error(f"AI error: {e}")


# ============================================================
# FEATURE 2: AI SUMMARIZATION
# ============================================================


with tab2:

    st.subheader("📝 AI Document Summarizer")

    st.write(
        "Generate accurate, structured and easy-to-understand "
        "summaries while preserving important information."
    )

    summary_style = st.selectbox(
        "Choose summary style",
        [
            "Short Summary",
            "Detailed Summary",
            "Student-Friendly Summary",
            "Research Paper Summary"
        ]
    )

    if st.button("✨ Generate Summary", key="summary_button"):

        try:

            document_text = get_full_document_text()

            if not document_text.strip():
                st.warning("No readable text found in the document.")
                st.stop()

            # Keep input within a practical API context budget
            document_text = document_text[:90000]

            with st.spinner("Analyzing document and generating summary..."):

                # Divide document into smaller sections
                parts = [
                    document_text[i:i + 10000]
                    for i in range(0, len(document_text), 10000)
                ]

                partial_summaries = []

                # Stage 1: Summarize individual sections
                for index, part in enumerate(parts):

                    partial = ask_groq(
                        f"""
You are analyzing section {index + 1} of a larger document.

Create an informative summary of this section.

IMPORTANT INSTRUCTIONS:

1. Preserve all important definitions and explanations.
2. Include important facts, technical terms and examples.
3. Preserve numerical values, names and findings.
4. Include algorithms, methods and processes mentioned.
5. Explain important concepts clearly.
6. Do not remove important information just to make
   the summary shorter.
7. Do not add information that is not in the source.
8. If a topic is not discussed, do not invent it.
9. Avoid unnecessary repetition.

SECTION CONTENT:

{part}

Write the section summary using clear headings and bullet points.
""",
                        system_prompt=(
                            "You are an accurate academic document "
                            "summarization assistant. Your priority is "
                            "preserving source information."
                        ),
                        api_key=api_key
                    )

                    partial_summaries.append(partial)

                # Stage 2: Combine all section summaries
                combined = "\n\n".join(partial_summaries)

                # Choose instructions based on selected style
                if summary_style == "Short Summary":

                    style_instruction = """
Create a concise summary containing the main topic,
important concepts, key findings and conclusions.
Keep essential definitions and facts.
"""

                elif summary_style == "Detailed Summary":

                    style_instruction = """
Create a comprehensive summary.

Include:
1. Main topic and introduction
2. All important concepts and definitions
3. Types and categories
4. Algorithms, methods and techniques
5. Processes and workflows
6. Important examples and explanations
7. Numerical values and findings
8. Applications
9. Important conclusions

Explain each important concept clearly.
Do not omit significant information.
"""

                elif summary_style == "Student-Friendly Summary":

                    style_instruction = """
Explain the document in simple English for students.

Include:
1. Main topic
2. Important definitions
3. Concepts explained in simple words
4. Easy examples from the source
5. Important algorithms and methods
6. Key points for examination
7. Conclusions

Make the content easy to understand without
changing the original meaning.
"""

                else:

                    style_instruction = """
Prepare an academic research-style summary.

Include:
1. Research topic
2. Background and motivation, if available
3. Objectives, if mentioned
4. Methods and techniques
5. Important findings
6. Results and observations
7. Applications and implications
8. Limitations, if mentioned
9. Conclusions

Do not invent research objectives, results or limitations
that are not present in the source.
"""

                final_summary = ask_groq(
                    f"""
You are preparing the final summary of a PDF document.

Selected summary style: {summary_style}

{style_instruction}

IMPORTANT RULES:

1. Use only information supported by the source summaries.
2. Do not invent facts or unsupported explanations.
3. Preserve important technical terms.
4. Do not confuse different concepts.
5. Avoid repeating the same information.
6. Use clear headings and organized bullet points.
7. Maintain the meaning of the original document.
8. Do not claim that the document contains information
   that is not present in the supplied summaries.

SECTION SUMMARIES:

{combined}

Generate the final summary now.
""",
                    system_prompt=(
                        "You are DocuMind AI, an expert academic "
                        "document summarizer. Produce accurate, "
                        "structured and readable summaries."
                    ),
                    api_key=api_key
                )

                st.session_state.summary = final_summary

            st.success("Summary generated successfully!")

        except Exception as e:

            st.error(f"Summary generation failed: {e}")

    if st.session_state.summary:

        st.markdown("### 📖 Generated Summary")

        st.markdown(
            '<div class="answer-card">',
            unsafe_allow_html=True
        )

        st.markdown(st.session_state.summary)

        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download Summary as TXT",
            data=st.session_state.summary,
            file_name="DocuMind_Summary.txt",
            mime="text/plain"
        )



# ============================================================
# FEATURE 3: QUIZ GENERATOR
# ============================================================

with tab3:

    st.subheader("🧩 AI Quiz Generator")

    st.write(
        "Generate multiple-choice questions from your uploaded PDF."
    )

    number_of_questions = st.slider(
        "Number of questions",
        min_value=3,
        max_value=10,
        value=5
    )

    difficulty = st.selectbox(
        "Select difficulty",
        ["Easy", "Medium", "Hard"]
    )

    if st.button("🚀 Generate Quiz", key="quiz_button"):

        try:

            document_text = get_full_document_text()[:45000]

            with st.spinner("Creating your quiz..."):

                quiz = ask_groq(
                    f"""
Create exactly {number_of_questions} multiple-choice
questions based ONLY on this document.

Difficulty: {difficulty}

For each question provide:
- Question number
- Question
- Four options (A, B, C, D)
- Correct answer
- Short explanation

Make sure every correct answer is supported
by the document.

DOCUMENT:
{document_text}
""",
                    system_prompt=(
                        "You are an educational quiz generator. "
                        "Create clear and accurate MCQs. "
                        "Never invent facts outside the document."
                    ),
                    api_key=api_key
                )

                st.session_state.quiz = quiz

        except Exception as e:

            st.error(f"Quiz generation failed: {e}")

    if st.session_state.quiz:

        st.markdown("### 🎯 Your Generated Quiz")

        st.markdown(st.session_state.quiz)

        st.download_button(
            "⬇️ Download Quiz",
            data=st.session_state.quiz,
            file_name="DocuMind_Quiz.txt",
            mime="text/plain"
        )


# ============================================================
# FEATURE 4: PDF REPORT GENERATOR
# ============================================================

with tab4:

    st.subheader("📑 AI PDF Report Generator")

    st.write(
        "Generate a structured report based on your uploaded document."
    )

    report_type = st.selectbox(
        "Choose report type",
        [
            "General Document Report",
            "Research Paper Analysis",
            "Study Notes Report",
            "Executive Summary Report"
        ]
    )

    if st.button("📄 Generate PDF Report", key="report_button"):

        try:

            document_text = get_full_document_text()[:60000]

            with st.spinner("Generating your report..."):

                report = ask_groq(
                    f"""
Create a professional {report_type}.

Use the uploaded document as the only factual source.

Include suitable headings such as:
1. Introduction
2. Main Topic
3. Important Concepts
4. Detailed Analysis
5. Key Findings
6. Conclusion

Use clear and professional English.
Do not invent unsupported information.

DOCUMENT:
{document_text}
""",
                    system_prompt=(
                        "You are a professional document report writer. "
                        "Create structured, accurate reports."
                    ),
                    api_key=api_key
                )

                st.session_state.report = report

        except Exception as e:

            st.error(f"Report generation failed: {e}")

    if st.session_state.report:

        st.markdown("### 📘 Generated Report")

        st.markdown(st.session_state.report)

        try:

            pdf_data = create_pdf_report(
                report_type,
                st.session_state.report
            )

            st.download_button(
                "⬇️ Download Professional PDF Report",
                data=pdf_data,
                file_name="DocuMind_AI_Report.pdf",
                mime="application/pdf"
            )

        except Exception as e:

            st.error(f"PDF creation error: {e}")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#9ca3af;">
        <b>DocuMind AI</b> | Powered by Generative AI and Groq
        <br>
        Intelligent Document Understanding System
    </div>
    """,
    unsafe_allow_html=True
)
