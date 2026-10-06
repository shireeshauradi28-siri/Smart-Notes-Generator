import streamlit as st
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Notes Generator",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Smart Notes Generator")
st.write("Upload a PDF or Word document and generate smart notes using AI.")

# -----------------------------
# Ollama Model
# -----------------------------
llm = OllamaLLM(model="llama3.2")

# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Extract text from DOCX
# -----------------------------
def extract_docx_text(file):
    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text + "\n"

    return text


# -----------------------------
# Generate Notes
# -----------------------------
def generate_notes(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    notes = []

    progress = st.progress(0)

    for i, chunk in enumerate(chunks):

        prompt = f"""
You are an expert student note-making assistant.

Create clear and easy-to-study notes from the following content.

Requirements:
- Give a suitable title
- Use headings and subheadings
- Use bullet points
- Highlight important concepts
- Include important definitions
- Include important formulas if present
- Include key points
- Keep the language simple
- Do not add information that is not present in the document

CONTENT:

{chunk}

Generate the notes now.
"""

        response = llm.invoke(prompt)
        notes.append(response)

        progress.progress((i + 1) / len(chunks))

    return "\n\n".join(notes)


# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your PDF or Word document",
    type=["pdf", "docx"]
)

# -----------------------------
# Process File
# -----------------------------
if uploaded_file is not None:

    st.success(f"File uploaded: {uploaded_file.name}")

    if st.button("🚀 Generate Smart Notes"):

        with st.spinner("Reading your document..."):

            if uploaded_file.name.lower().endswith(".pdf"):
                text = extract_pdf_text(uploaded_file)

            elif uploaded_file.name.lower().endswith(".docx"):
                text = extract_docx_text(uploaded_file)

            else:
                text = ""

        if not text.strip():

            st.error(
                "Could not extract text from this document. "
                "Please upload a text-based PDF or DOCX file."
            )

        else:

            st.info("Generating smart notes using Llama 3.2...")

            notes = generate_notes(text)

            st.success("✅ Smart notes generated successfully!")

            st.markdown("## 📚 Your Smart Notes")

            st.markdown(notes)

            st.download_button(
                label="📥 Download Notes",
                data=notes,
                file_name="smart_notes.txt",
                mime="text/plain"
            )