# import os
# import re
# from typing import List, Dict, Optional
# from pypdf import PdfReader
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv

# import streamlit as st

# load_dotenv()

# class ChecklistRulesExtractor:
#     """Extracts structured checklist rules from a tabular-style PDF checklist."""

#     def __init__(self, pdf_path: Optional[str] = None):
#         self.pdf_path = pdf_path or os.getenv("CHECKLIST_PDF", st.secrets.get("CHECKLIST_PDF"))
#         if not self.pdf_path:
#             raise ValueError("Checklist PDF path not provided.")
#         if not os.path.exists(self.pdf_path):
#             raise FileNotFoundError(f"Checklist PDF not found at: {self.pdf_path}")

#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )

#     def extract_raw_text(self) -> str:
#         """Extract all text from the PDF file."""
#         try:
#             with open(self.pdf_path, 'rb') as f:
#                 reader = PdfReader(f)
#                 return "\n".join([page.extract_text() or "" for page in reader.pages])
#         except Exception as e:
#             raise RuntimeError(f"Failed to read checklist PDF: {e}")

#     def extract_rules(self) -> Dict[str, List[str]]:
#         """Parses checklist questions and rationales grouped by category."""
#         text = self.extract_raw_text()

#         # Match category headers like '1.1 Preamble'
#         category_pattern = re.compile(r'(\d+\.\d+)\s+([^\n]+)')
#         categories = {match[0]: match[1].strip() for match in category_pattern.findall(text)}

#         # Match full rule entries like: '1.1.1 Question\nReason'
#         rule_pattern = re.compile(r'(\d+\.\d+\.\d+)\s+(.+?)\n\s*(.+?)(?=\n\d+\.\d+\.\d+|\n\d+\.\d+|\Z)', re.DOTALL)
#         rules_by_category: Dict[str, List[str]] = {}

#         for number, question, reason in rule_pattern.findall(text):
#             category_key = ".".join(number.split('.')[:2])
#             category_name = categories.get(category_key, f"Section {category_key}")

#             # Cleanup formatting
#             cleaned_q = re.sub(r'\s+', ' ', question).strip()
#             cleaned_r = re.sub(r'\s+', ' ', reason).strip()

#             combined = f"{number} — {cleaned_q}\nWhy: {cleaned_r}"

#             if category_name not in rules_by_category:
#                 rules_by_category[category_name] = []
#             rules_by_category[category_name].append(combined)

#         return rules_by_category

#     def get_rules_as_documents(self) -> List[Document]:
#         """Returns structured rules as LangChain Document objects."""
#         rules_by_category = self.extract_rules()
#         documents = []

#         for category, rules in rules_by_category.items():
#             for rule in rules:
#                 documents.append(Document(
#                     page_content=rule,
#                     metadata={
#                         "category": category,
#                         "source": self.pdf_path,
#                         "rule_type": "writing_checklist"
#                     }
#                 ))

#         return self.text_splitter.split_documents(documents)

#     def get_formatted_rules(self) -> str:
#         """Returns all checklist rules formatted as markdown."""
#         rules_by_category = self.extract_rules()
#         markdown = ""

#         for category, rules in rules_by_category.items():
#             markdown += f"### {category}\n"
#             for rule in rules:
#                 markdown += f"- {rule}\n"
#             markdown += "\n"

#         return markdown


import os
import re
from typing import List, Dict, Optional
from pypdf import PdfReader
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing import Optional, List, Dict
import os, re
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class ChecklistRulesExtractor:
    """Extracts structured checklist rules from a multi-line or merged-line PDF layout."""

    def __init__(self, pdf_path: Optional[str] = None):
        self.pdf_path = pdf_path or st.secrets.get("CHECKLIST_PDF") or os.getenv("CHECKLIST_PDF")

        if not self.pdf_path:
            st.error("❌ Checklist PDF path not provided (via env or secrets).")
            raise ValueError("Checklist PDF path not provided.")

        st.write(f"📄 Path resolved to: `{self.pdf_path}`")
        st.write(f"📂 Working directory: `{os.getcwd()}`")

        if not os.path.exists(self.pdf_path):
            files = []
            for root, _, filenames in os.walk("."):
                for fname in filenames:
                    files.append(os.path.join(root, fname))
            st.error(f"🚫 File not found: `{self.pdf_path}`")
            st.write("🔍 Available files in project:", files)
            raise FileNotFoundError(f"Checklist PDF not found at: {self.pdf_path}")

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def extract_raw_text(self) -> str:
        """Extract raw text from PDF pages using pypdf."""
        try:
            with open(self.pdf_path, 'rb') as f:
                reader = PdfReader(f)
                texts = []
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        texts.append(page_text)
                    else:
                        st.warning(f"🔇 No text on page {i}")
                raw_text = "\n".join(texts)
                st.write("📑 Raw text preview:", raw_text[:300])
                return raw_text
        except Exception as e:
            st.error(f"❌ PDF read failed: {e}")
            raise RuntimeError(f"Failed to read checklist PDF: {e}")

    def extract_rules(self) -> Dict[str, List[str]]:
        """Parses checklist questions and rationales grouped by section headers, with merged-line support."""
        text = self.extract_raw_text()
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Extract section headers like '1.1 Preamble'
        category_map = {}
        for i in range(len(lines)):
            match = re.match(r'^(\d+\.\d+)\s+(.*)', lines[i])
            if match:
                category_map[match.group(1)] = match.group(2).strip()

        rules_by_category: Dict[str, List[str]] = {}
        i = 0
        while i < len(lines):
            if re.match(r'^\d+\.\d+\.\d+', lines[i]):
                rule_line = lines[i]
                rule_id_match = re.match(r'^(\d+\.\d+\.\d+)\s+(.*)', rule_line)
                if rule_id_match:
                    rule_id = rule_id_match.group(1)
                    rest = rule_id_match.group(2)
                    # Heuristic split: look for punctuation followed by capital letter
                    parts = re.split(r'(?<=[\?\.\:])\s+(?=[A-Z])', rest)
                    question = parts[0] if parts else ""
                    reason = parts[1] if len(parts) > 1 else ""
                    category_key = ".".join(rule_id.split('.')[:2])
                    category_name = category_map.get(category_key, f"Section {category_key}")
                    combined = f"{rule_id} — {question}\nWhy: {reason}"
                    rules_by_category.setdefault(category_name, []).append(combined)
                i += 1
            else:
                i += 1

        st.write(f"🔎 Total rules parsed: {sum(len(v) for v in rules_by_category.values())}")
        st.success(f"✅ Extracted rules from {len(rules_by_category)} categories")
        return rules_by_category

    def get_rules_as_documents(self) -> List[Document]:
        """Returns parsed rules as LangChain-compatible Document objects."""
        rules_by_category = self.extract_rules()
        documents = []
        for category, rules in rules_by_category.items():
            for rule in rules:
                documents.append(Document(
                    page_content=rule,
                    metadata={
                        "category": category,
                        "source": self.pdf_path,
                        "rule_type": "writing_checklist"
                    }
                ))
        return self.text_splitter.split_documents(documents)

    def get_formatted_rules(self) -> str:
        """Returns all rules formatted as Markdown."""
        rules_by_category = self.extract_rules()
        markdown = ""
        for category, rules in rules_by_category.items():
            markdown += f"### {category}\n"
            for rule in rules:
                markdown += f"- {rule}\n"
            markdown += "\n"
        return markdown

