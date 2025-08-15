# import os
# import sys
# import tempfile
# import shutil
# from typing import List
# from io import BytesIO

# import streamlit as st
# from dotenv import load_dotenv
# from langchain.schema import Document

# # Add the "src" folder to sys.path
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = os.path.join(CURRENT_DIR, "src")
# if SRC_DIR not in sys.path:
#     sys.path.insert(0, SRC_DIR)

# # Load environment variables early
# load_dotenv()

# # ✅ Use absolute imports from src (now in path)
# try:
#     from rag_agent import RAGAgent
#     from nvidia_embeddings import NVIDIAEmbeddings
#     from llm_hf import HuggingFaceLLM
# except ImportError as e:
#     st.error(f"❌ Import error: {e}")
#     st.stop()


# def initialize_session():
#     """Initialize the RAG agent and embeddings"""
#     if "rag_agent" not in st.session_state:
#         try:
#             # Test NVIDIA embedding API connection
#             if not NVIDIAEmbeddings().test_connection():
#                 st.error("🚫 Failed to connect to NVIDIA Embedding API")
#                 st.stop()

#             rag_agent = RAGAgent()
#             if not rag_agent.load_checklist():
#                 st.error("🚫 Failed to load checklist rules")
#                 st.stop()

#             st.session_state.rag_agent = rag_agent
#             st.session_state.processed = False
#             st.session_state.combined_output = None

#         except Exception as e:
#             st.error(f"🚨 Initialization failed: {str(e)}")
#             st.stop()


# def save_uploaded_files(files: List[BytesIO]) -> str:
#     """Save uploaded PDFs to a temp directory and return the path"""
#     temp_dir = tempfile.mkdtemp()
#     for file in files:
#         file_path = os.path.join(temp_dir, file.name)
#         try:
#             with open(file_path, "wb") as f:
#                 f.write(file.getbuffer())
#         except Exception as e:
#             st.error(f"❌ Could not save {file.name}: {e}")
#     return temp_dir


# def main():
#     st.set_page_config(
#         page_title="NVIDIA RAG Document Rewriter",
#         layout="wide",
#         page_icon="🧠"
#     )

#     st.title("📝 NVIDIA RAG Document Rewriter")
#     st.markdown("Upload documents and rewrite them according to your internal writing checklist.")

#     initialize_session()

#     # Sidebar for configuration and file upload
#     with st.sidebar:
#         st.subheader("⚙️ Configuration")
#         with st.expander("Advanced Settings"):
#             temperature = st.slider("Creativity", 0.0, 1.0, 0.3)
#             similarity_threshold = st.slider("Rule Strictness", 0.5, 1.0, 0.7)

#         st.subheader("📄 Upload PDFs")
#         uploaded_files = st.file_uploader(
#             "Upload your PDF documents",
#             type=["pdf"],
#             accept_multiple_files=True
#         )

#         if st.button("🚀 Process Documents") and uploaded_files:
#             with st.spinner("Analyzing documents..."):
#                 temp_dir = save_uploaded_files(uploaded_files)
#                 try:
#                     docs = st.session_state.rag_agent.load_documents(temp_dir)
#                     st.session_state.rag_agent.create_document_index(docs)

#                     # Combine all chunk content
#                     combined_text = "\n\n".join(doc.page_content for doc in docs)
#                     combined_document = Document(page_content=combined_text, metadata={"filename": "combined.md"})

#                     # Rewrite once using full checklist
#                     output = st.session_state.rag_agent.generate_rewritten_content(combined_document)
#                     st.session_state.combined_output = output
#                     st.session_state.processed = True

#                     st.success(f"✅ Rewritten combined document generated successfully!")

#                 except Exception as e:
#                     st.error(f"⚠️ Processing failed: {e}")
#                 finally:
#                     shutil.rmtree(temp_dir, ignore_errors=True)

#     # Main output section
#     if st.session_state.get("processed") and st.session_state.combined_output:
#         st.subheader("🧾 Rewritten Combined Document")
#         st.download_button(
#             label="💾 Download Rewritten Markdown",
#             data=st.session_state.combined_output,
#             file_name="rewritten_combined.md",
#             mime="text/markdown",
#             key="rewritten_single_doc"
#         )
#         st.markdown(st.session_state.combined_output)

#         st.subheader("🐞 Debug Info")
#         st.json({
#             "embedding_model": "nvidia/llama-nemoretriever-colembed-3b-v1",
#             "llm_model": os.getenv("LLM_MODEL", "unknown"),
#             "document_chunks_processed": len(st.session_state.rag_agent.doc_chunks),
#             "combined_mode": True
#         })


# if __name__ == "__main__":
#     main()







import os
import sys
import tempfile
import shutil
from typing import List
from io import BytesIO
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from typing import List, Dict

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

import streamlit as st

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO
from typing import List, Dict


if "timestamped_script" not in st.session_state:
    st.session_state.timestamped_script = []



# Add the "src" folder to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Load environment variables early
load_dotenv()

# ✅ Use absolute imports from src (now in path)
try:
    from rag_agent import RAGAgent
    from nvidia_embeddings import NVIDIAEmbeddings
    from llm_hf import HuggingFaceLLM
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()


from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import simpleSplit
from io import BytesIO

def convert_text_to_pdf(text: str) -> BytesIO:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER
    y = height - 40

    lines = text.split('\n')
    styles = getSampleStyleSheet()
    font_size = 11
    line_spacing = font_size + 4

    pdf.setFont("Helvetica", font_size)

    for line in lines:
        line = line.strip()
        if not line:
            y -= line_spacing
            continue

        # Basic formatting
        if line.startswith("# "):
            pdf.setFont("Helvetica-Bold", font_size + 2)
            line = line[2:]
        elif line.startswith("## "):
            pdf.setFont("Helvetica-Bold", font_size + 1)
            line = line[3:]
        elif line.startswith("- ") or line.startswith("* "):
            line = u"\u2022 " + line[2:]  # bullet point
            pdf.setFont("Helvetica", font_size)
        else:
            pdf.setFont("Helvetica", font_size)

        # Line wrapping
        wrapped = simpleSplit(line, "Helvetica", font_size, width - 80)
        for wrapped_line in wrapped:
            if y < 50:
                pdf.showPage()
                y = height - 40
                pdf.setFont("Helvetica", font_size)
            pdf.drawString(50, y, wrapped_line)
            y -= line_spacing

    pdf.save()
    buffer.seek(0)
    return buffer


def initialize_session():
    """Initialize the RAG agent and embeddings"""
    if "rag_agent" not in st.session_state:
        try:
            if not NVIDIAEmbeddings().test_connection():
                st.error("🚫 Failed to connect to NVIDIA Embedding API")
                st.stop()

            rag_agent = RAGAgent()
            if not rag_agent.load_checklist():
                st.error("🚫 Failed to load checklist rules")
                st.stop()

            st.session_state.rag_agent = rag_agent
            st.session_state.processed = False
            st.session_state.combined_output = None
            st.session_state.final_output = None

        except Exception as e:
            st.error(f"🚨 Initialization failed: {str(e)}")
            st.stop()


def save_uploaded_files(files: List[BytesIO]) -> str:
    """Save uploaded PDFs to a temp directory and return the path"""
    temp_dir = tempfile.mkdtemp()
    for file in files:
        file_path = os.path.join(temp_dir, file.name)
        try:
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        except Exception as e:
            st.error(f"❌ Could not save {file.name}: {e}")
    return temp_dir

import re

# def escape_latex(text: str) -> str:
#     """Escape LaTeX-sensitive characters."""
#     replacements = {
#         '\\': r'\textbackslash{}',
#         '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
#         '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
#         '^': r'\textasciicircum{}'
#     }
#     return ''.join(replacements.get(char, char) for char in text)

# def convert_text_to_beamer_slides(content: str) -> str:
#     beamer_template = [
#         r"\documentclass{beamer}",
#         r"\usetheme{Madrid}",
#         r"\title{AI Rewritten Presentation}",
#         r"\author{Generated by NVIDIA RAG Rewriter}",
#         r"\date{\today}",
#         r"\begin{document}",
#         r"\frame{\titlepage}"
#     ]

#     lines = content.splitlines()
#     slide_title = None
#     slide_items = []

#     def flush_slide(title, items):
#         if title:
#             beamer_template.append(r"\begin{frame}{%s}" % escape_latex(title))
#             if items:
#                 beamer_template.append(r"\begin{itemize}")
#                 for item in items:
#                     beamer_template.append(r"\item %s" % escape_latex(item.strip()))
#                 beamer_template.append(r"\end{itemize}")
#             beamer_template.append(r"\end{frame}")

#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         if re.match(r"^#{1,3}\s+", line):  # Matches #, ##, ### headings
#             if slide_title:
#                 flush_slide(slide_title, slide_items)
#             slide_title = re.sub(r"^#{1,3}\s+", "", line)
#             slide_items = []
#         elif line.startswith("- ") or line.startswith("* "):
#             slide_items.append(line[2:].strip())

#     flush_slide(slide_title, slide_items)

#     beamer_template.append(r"\end{document}")
#     return "\n".join(beamer_template)


import re

def escape_latex(text: str) -> str:
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
        '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}'
    }
    return ''.join(replacements.get(char, char) for char in text)

def clean_line(line: str) -> str:
    line = re.sub(r'<br\s*/?>', '', line, flags=re.IGNORECASE)
    return line.strip()

def convert_text_to_beamer_slides(content: str) -> str:
    beamer_template = [
        r"\documentclass[17pt]{beamer}",
        r"\usepackage[x11names]{xcolor}",
        r"\usepackage[colorlinks,",
        r"            citecolor=DeepPink4,",
        r"            linkcolor=white,",
        r"            urlcolor=DarkBlue]{hyperref}",
        r"\setbeamersize{text margin left=0.25cm,text margin right=0.5cm}",
        r"\usepackage{amsmath}",
        r"\usepackage{metalogo}",
        r"\usepackage{framed}",
        r"\usepackage{multicol}",
        r"\definecolor{PowderBlue}{RGB}{90,70,172}",
        r"\definecolor{Color1}{RGB}{138,128,33}",
        r"\usepackage{beamerthemesplit}",
        r"\setbeamercolor{structure}{fg=PowderBlue}",
        r"\setbeamercolor{alerted text}{fg=PowderBlue}",
        r"\logo{%",
        r"  \includegraphics[height=0.4 cm]{images/logo.png} \hspace{6.5cm}%",
        r"  \includegraphics[scale=0.2]{images/logo.png}\\[0.2cm]%",
        r"  \hspace{255pt} \includegraphics[scale=0.08]{images/st-logo-new.png}%",
        r"}",
        r"\begin{document}",
        r"\sffamily \bfseries",
        r"\title[\scriptsize {Auto-generated Slides}]{{\large Auto-generated Slides}}",
        r"\author[]{{\small Generated by NVIDIA RAG Rewriter \\ {\color{blue}https://spoken-tutorial.org } \\ [0.5cm] Script: ADITYA \\ Video: AI Copilot \\ [0.5cm] } {\small \today }}",
        r"\date{}",
        r"\begin{frame}",
        r"\titlepage",
        r"\end{frame}"
    ]

    lines = content.splitlines()
    slide_title = None
    slide_items = []
    slide_paragraph = []

    def flush_slide(title, items, paragraph):
        if title:
            beamer_template.append(r"\begin{frame}")
            beamer_template.append(r"\frametitle{%s} \pause" % escape_latex(title))
            if items:
                beamer_template.append(r"\begin{itemize}[<+-|alert@+>]")
                for item in items:
                    beamer_template.append(r"\item %s" % escape_latex(item))
                beamer_template.append(r"\end{itemize}")
            elif paragraph:
                for para in paragraph:
                    beamer_template.append(r"\vspace{0.2cm}")
                    beamer_template.append(escape_latex(para))
            beamer_template.append(r"\end{frame}")

    for line in lines:
        line = clean_line(line)
        if not line:
            continue
        if re.match(r"^#{1,3}\s+", line):  # Heading
            if slide_title:
                flush_slide(slide_title, slide_items, slide_paragraph)
            slide_title = re.sub(r"^#{1,3}\s+", "", line)
            slide_items = []
            slide_paragraph = []
        elif re.match(r"^[-*]\s+", line):  # Bullet point
            slide_items.append(re.sub(r"^[-*]\s+", "", line))
        else:
            slide_paragraph.append(line)

    flush_slide(slide_title, slide_items, slide_paragraph)

    beamer_template.append(r"\begin{frame}")
    beamer_template.append(r"\begin{center}")
    beamer_template.append(r"Thank you")
    beamer_template.append(r"\end{center}")
    beamer_template.append(r"\end{frame}")
    beamer_template.append(r"\end{document}")

    return "\n".join(beamer_template)




#For Downloading the Latex Beamer File with Logos and images

import os
import shutil
import subprocess

def prepare_beamer_project(content: str, image_paths: list, output_dir: str = "beamer_project", tex_filename: str = "presentation.tex") -> str:
    """
    Saves LaTeX code and copies images into a structured folder.
    """
    # Create output folders
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Save LaTeX file
    tex_path = os.path.join(output_dir, tex_filename)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(convert_text_to_beamer_slides(content))

    # Copy images
    for img_path in image_paths:
        shutil.copy(img_path, os.path.join(images_dir, os.path.basename(img_path)))

    return tex_path

def compile_latex_to_pdf(tex_path: str) -> str:
    """
    Compiles the LaTeX file to PDF using pdflatex.
    """
    output_dir = os.path.dirname(tex_path)
    tex_filename = os.path.basename(tex_path)

    # Run pdflatex twice for proper references
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_filename],
            cwd=output_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    pdf_path = os.path.join(output_dir, tex_filename.replace(".tex", ".pdf"))
    with open(pdf_path, "rb") as f:
        st.download_button("Download PDF Slides", f, file_name="presentation.pdf")
    return pdf_path


###############################



def parse_rewritten_text_to_script(content: str) -> List[Dict[str, str]]:
    script = []

    # Markdown table format
    if "Visual Cue | Narration" in content and "|" in content:
        lines = content.strip().split("\n")
        for line in lines[2:]:  # Skip header and separator
            if "|" in line:
                parts = [cell.strip() for cell in line.split("|")]
                if len(parts) >= 2:
                    script.append({
                        "Visual Cue": parts[0],
                        "Narration": parts[1]
                    })
        return script

    # Heading-based fallback
    current_heading = None
    current_body = []
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_heading and current_body:
                script.append({
                    "Visual Cue": current_heading,
                    "Narration": "\n".join(current_body).strip()
                })
                current_body = []
            current_heading = stripped.lstrip("#").strip()
        elif stripped:
            current_body.append(stripped)

    if current_heading and current_body:
        script.append({
            "Visual Cue": current_heading,
            "Narration": "\n".join(current_body).strip()
        })

    return script

from typing import List, Dict




def generate_timestamps_for_blocks(
    script: List[Dict[str, str]],
    start_time: int = 1,
    words_per_second: float = 2.3,
    min_duration: int = 2
) -> List[Dict[str, str]]:
    """
    Generate timestamped narration blocks with estimated durations.

    Args:
        script: List of narration blocks with 'Narration' and optional 'Visual Cue'.
        start_time: Initial timestamp in seconds.
        words_per_second: Speaking rate used to estimate duration.
        min_duration: Minimum duration per block in seconds.

    Returns:
        List of dicts with 'Start Time', 'End Time', 'Narration', and 'Visual Cue'.
    """
    timed_script = []
    current_time = start_time

    for block in script:
        narration = block.get("Narration") or block.get("narration", "")
        visual_cue = block.get("Visual Cue") or block.get("visual_cue", "")

        narration = str(narration).strip()
        visual_cue = str(visual_cue).strip()

        if not narration:
            continue

        word_count = len(narration.split())
        duration = max(round(word_count / words_per_second), min_duration)

        start_mm, start_ss = divmod(current_time, 60)
        end_time = current_time + duration
        end_mm, end_ss = divmod(end_time, 60)

        timed_script.append({
            "Start Time": f"{start_mm:02d}:{start_ss:02d}",
            "End Time": f"{end_mm:02d}:{end_ss:02d}",
            "Narration": narration,
            "Visual Cue": visual_cue
        })

        current_time = end_time

    return timed_script










from typing import List, Dict
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import re



def clean_markdown(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = re.sub(r"^\s*#+\s*", "", line)              # Remove headers
        line = re.sub(r"^\s*[-*|]\s*", "• ", line)          # Bullets or pipes to •
        line = re.sub(r"[*_`~|]", "", line)                 # Strip markdown symbols
        line = line.strip()
        if line:
            cleaned.append(line)
    return "<br/>".join(cleaned)

def convert_script_to_pdf(script: List[Dict[str, str]]) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        rightMargin=40,
        leftMargin=40,
        topMargin=60,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    cue_style = ParagraphStyle(
        name="CueStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.darkblue,
        spaceAfter=6
    )

    narration_style = ParagraphStyle(
        name="NarrationStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )

    elements = [Paragraph("🎞️ Slide-by-Slide Narration Script", styles["Title"]), Spacer(1, 20)]
    table_data = [["Visual Cue", "Narration"]]

    for i, block in enumerate(script):
        cue_raw = block.get("Visual Cue") or block.get("visual_cue", "")
        narration_raw = block.get("Narration") or block.get("narration", "")

        cue_heading = f"<b>Slide {i+1} Title:</b> {cue_raw}"
        narration_cleaned = clean_markdown(narration_raw)

        cue_paragraph = Paragraph(cue_heading, cue_style)
        narration_paragraph = Paragraph(narration_cleaned, narration_style)

        table_data.append([cue_paragraph, narration_paragraph])

    table = Table(table_data, colWidths=[160, 360])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightyellow])
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.set_page_config(
    page_title="Spoken Tutorial-IIT Bombay",
    layout="wide",
    page_icon="🧠"
)




from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors





def main():
    st.title("📝 Spoken Tutorial - Automation RAG")
    st.markdown("Upload documents and rewrite them using your internal checklist and AI rewriting assistant.")
    initialize_session()

    # ✅ Safe session state initialization
    for key in ["processed", "combined_output", "final_output", "tutorial_script", "timestamped_script", "edited_script"]:
        if key not in st.session_state:
            st.session_state[key] = [] if "script" in key else ""

    # Sidebar configuration
    with st.sidebar:
        st.subheader("⚙️ Configuration")
        with st.expander("Advanced Settings"):
            temperature = st.slider("Creativity", 0.0, 1.0, 0.3)
            similarity_threshold = st.slider("Rule Strictness", 0.5, 1.0, 0.7)

        st.subheader("📄 Upload PDFs")
        uploaded_files = st.file_uploader("Upload your PDF documents", type=["pdf"], accept_multiple_files=True)








        if st.button("🚀 Process Documents") and uploaded_files:
            with st.spinner("Analyzing documents..."):
                temp_dir = save_uploaded_files(uploaded_files)
                try:
                    docs = st.session_state.rag_agent.load_documents(temp_dir)
                    st.session_state.rag_agent.create_document_index(docs)

                    combined_text = "\n\n".join(doc.page_content for doc in docs)
                    combined_document = Document(page_content=combined_text, metadata={"filename": "combined.md"})

                    output = st.session_state.rag_agent.generate_rewritten_content(combined_document)
                    st.session_state.combined_output = output
                    st.session_state.processed = True
                    st.success("✅ Rewritten combined document generated successfully!")
                except Exception as e:
                    st.error(f"⚠️ Processing failed: {e}")
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

    # Combined document editing
    if st.session_state.processed and st.session_state.combined_output:
        st.subheader("🧾 Rewritten Combined Document")
        st.download_button("💾 Download Original Rewrite", data=st.session_state.combined_output,
                           file_name="rewritten_combined.md", mime="text/markdown")

        st.markdown("### ✍️ Edit or Refine the Output Below")
        edited_text = st.text_area("Modify the content if needed:", value=st.session_state.combined_output, height=400)

        if st.button("🔄 Regenerate Based on Edits"):
            with st.spinner("Reprocessing modified input..."):
                try:
                    updated_doc = Document(page_content=edited_text, metadata={"source": "edited_user_text"})
                    new_output = st.session_state.rag_agent.generate_rewritten_content(updated_doc)
                    st.session_state.final_output = new_output
                    st.session_state.tutorial_script = parse_rewritten_text_to_script(new_output)
                    st.session_state.timestamped_script = generate_timestamps_for_blocks(st.session_state.tutorial_script)
                    st.success("✅ Regenerated output from your edited content.")
                except Exception as e:
                    st.error(f"❌ Regeneration failed: {e}")

    # Slide-by-slide editing interface
    if st.session_state.timestamped_script:
        st.subheader("🎙️ Edit Slide Narration")
        edited_script = []
        for i, block in enumerate(st.session_state.timestamped_script):
            with st.expander(f"🎬 Slide {i+1}: {block.get('Visual Cue', f'Slide {i+1}')}", expanded=False):
                cue = block.get("Visual Cue", f"Slide {i+1}")
                narration = block.get("Narration", "")
                st.markdown(f"**Visual Cue:** {cue}")
                manual_edit = st.text_area("✏️ Manual Edit", value=narration, key=f"manual_{i}")
                refine_hint = st.text_input("💬 LLM Hint (Optional)", key=f"hint_{i}", placeholder="Make it concise...")

                final_text = manual_edit
                if st.button("🤖 Refine with LLM", key=f"refine_{i}") and refine_hint:
                    checklist_rules = "\n".join(
                        f"- {doc.page_content}" for doc in st.session_state.rag_agent.retriever.get_relevant_documents(manual_edit[:1000])
                    )
                    prompt = f"""Checklist Rules:
{checklist_rules}

Original Narration:
{manual_edit}

User Request:
{refine_hint}

Return narration with clean formatting, no markdown symbols, and bullet clarity."""
                    refined = st.session_state.rag_agent.llm.generate(prompt)
                    st.text_area("🧠 LLM Suggestion", value=refined, key=f"suggestion_{i}", height=160)
                    final_text = refined

                cleaned = clean_markdown(final_text)
                edited_script.append({"Visual Cue": cue, "Narration": cleaned})

        st.session_state.edited_script = edited_script

        # 🔄 Regenerate from edited narration blocks
        st.subheader("🔄 Regenerate Based on Edited Narration Blocks")
        if st.button("♻️ Regenerate from Edited Slides"):
            with st.spinner("Regenerating full structure and timestamps..."):
                try:
                    combined_text = "\n\n".join(
                        f"# {block['Visual Cue']}\n{block['Narration']}" for block in st.session_state.edited_script
                    )
                    updated_doc = Document(page_content=combined_text, metadata={"source": "edited_script_blocks"})

                    regenerated_output = st.session_state.rag_agent.generate_rewritten_content(updated_doc)
                    st.session_state.final_output = regenerated_output
                    st.session_state.tutorial_script = parse_rewritten_text_to_script(regenerated_output)
                    st.session_state.timestamped_script = generate_timestamps_for_blocks(st.session_state.tutorial_script)
                    st.session_state.edited_script = parse_rewritten_text_to_script(regenerated_output)

                    with st.expander("🐞 Debug Snapshot"):
                        st.json({
        "final_output": bool(st.session_state.get("final_output")),
        "edited_script_len": len(st.session_state.get("edited_script", [])),
        "timestamped_script_len": len(st.session_state.get("timestamped_script", []))
    })



                    st.success("✅ Regenerated output based on edited slides.")
                    
                    with st.expander("🐞 Debug Snapshot"):
                        st.json({
                    "parsed_blocks": st.session_state.tutorial_script,
                    "timestamped_script": st.session_state.timestamped_script
                })
                    
                    
                except Exception as e:
                    st.error(f"❌ Failed to regenerate from edited slides: {e}")
        # ✅ Preview raw regenerated markdown
        if st.session_state.final_output:
            st.subheader("📝 Final Output Preview")
            st.markdown(st.session_state.final_output)

# ✅ Check if edited_script got parsed
        if st.session_state.edited_script and len(st.session_state.edited_script) > 0:
            st.success(f"🎯 {len(st.session_state.edited_script)} narration blocks loaded.")
            st.markdown("### ✨ First Block Preview")
            first_block = st.session_state.edited_script[0]
            st.markdown(f"#### {first_block['Visual Cue']}")
            st.markdown(first_block["Narration"])
        else:
            st.warning("⚠️ No narration blocks parsed — check if output contains valid headings (# Slide Title) and bullets.")

        # Export narration script
        st.subheader("📥 Export Final Narration Script")
        final_pdf = convert_script_to_pdf(st.session_state.edited_script)
        st.download_button("📄 Download Final Narration PDF", data=final_pdf,
                           file_name="final_narration_script.pdf", mime="application/pdf")
        final_csv = pd.DataFrame(st.session_state.edited_script).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Final Script as CSV", data=final_csv,
                           file_name="final_narration_script.csv", mime="text/csv")
        # ✅ Generate LaTeX Beamer code from cleaned narration
        beamer_markdown = "\n\n".join(
            f"# {block['Visual Cue']}\n" + "\n".join(f"- {line.strip()}" for line in block["Narration"].split("•") if line.strip())
            for block in st.session_state.edited_script
            )
        beamer_code = convert_text_to_beamer_slides(beamer_markdown)

# ✅ View generated .tex code
        st.subheader("📄 LaTeX Beamer (.tex) Preview")
        with st.expander("🔍 View Generated .tex Code"):
            st.code(beamer_code, language="latex")

# ✅ Editable LaTeX export window
        st.subheader("📝 Edit LaTeX (.tex) Code Before Download")
        edited_tex = st.text_area("Modify the .tex code here:", value=beamer_code, height=500)
        st.download_button("📎 Download Edited .tex", data=edited_tex, file_name="custom_slides.tex", mime="text/plain")


        st.subheader("🎞️ Slide Preview Navigator")
        slide_index = st.slider("Slide Number", 1, len(st.session_state.edited_script), 1)
        slide = st.session_state.edited_script[slide_index - 1]
        st.markdown(f"### 📌 Slide {slide_index}: {slide['Visual Cue']}")
        for bullet in slide["Narration"].split("•"):
            if bullet.strip():
                st.markdown(f"- {bullet.strip()}")

        uploaded_image = st.file_uploader("📷 Add Image to This Slide", type=["png", "jpg"], key=f"img_{slide_index}")
        if uploaded_image:
            st.image(uploaded_image, caption="Slide Visual", use_column_width=True)
            st.session_state[f"slide_img_{slide_index}"] = uploaded_image





        if st.session_state.timestamped_script:
            st.subheader("⏱️ Timestamped Narration")
            ts_df = pd.DataFrame(st.session_state.timestamped_script)
            st.dataframe(ts_df, use_container_width=True)

            # CSV Export
            ts_csv = ts_df.to_csv(index=False).encode("utf-8")
            st.download_button("🕒 Download Timestamped Script (CSV)", data=ts_csv,
                            file_name="timestamped_script.csv", mime="text/csv")

            # PDF Export
            styles = getSampleStyleSheet()
            narration_style = ParagraphStyle(
                name="NarrationStyle",
                parent=styles["Normal"],
                fontSize=10,
                leading=14,
                spaceAfter=6
            )

            time_style = ParagraphStyle(
                name="TimeStyle",
                parent=styles["Normal"],
                fontSize=10,
                leading=14,
                textColor=colors.darkblue,
                spaceAfter=6
            )

            table_data = [["Start Time", "End Time", "Narration"]]

            for row in st.session_state.timestamped_script:
                start = Paragraph(f"<b>{row['Start Time']}</b>", time_style)
                end = Paragraph(f"<b>{row['End Time']}</b>", time_style)
                narration = Paragraph(clean_markdown(row["Narration"]), narration_style)
                table_data.append([start, end, narration])

            ts_pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(ts_pdf_buffer, pagesize=LETTER,
                                    rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)

            table = Table(table_data, colWidths=[70, 70, 360])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightyellow])
            ]))

            doc.build([table])
            ts_pdf_buffer.seek(0)
            st.download_button("📘 Download Timestamped Script (PDF)", data=ts_pdf_buffer,
                            file_name="timestamped_script.pdf", mime="application/pdf")






    # 🐞 Debug Info
    st.subheader("🐞 Debug Info")
    st.json({
        "embedding_model": "nvidia/llama-nemoretriever-colembed-3b-v1",
        "llm_model": os.getenv("LLM_MODEL", "unknown"),
        "document_chunks_processed": len(st.session_state.rag_agent.doc_chunks),
        "combined_mode": True
    })



if __name__ == "__main__":
    main()
