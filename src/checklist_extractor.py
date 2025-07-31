import os
import re
from typing import List, Dict, Optional
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

import streamlit as st

load_dotenv()

class ChecklistRulesExtractor:
    """Extracts structured checklist rules from a tabular-style PDF checklist."""

    def __init__(self, pdf_path: Optional[str] = None):
        self.pdf_path = pdf_path or os.getenv("CHECKLIST_PDF", st.secrets.get("CHECKLIST_PDF"))
        if not self.pdf_path:
            raise ValueError("Checklist PDF path not provided.")
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"Checklist PDF not found at: {self.pdf_path}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def extract_raw_text(self) -> str:
        """Extract all text from the PDF file."""
        try:
            with open(self.pdf_path, 'rb') as f:
                reader = PdfReader(f)
                return "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            raise RuntimeError(f"Failed to read checklist PDF: {e}")

    def extract_rules(self) -> Dict[str, List[str]]:
        """Parses checklist questions and rationales grouped by category."""
        text = self.extract_raw_text()

        # Match category headers like '1.1 Preamble'
        category_pattern = re.compile(r'(\d+\.\d+)\s+([^\n]+)')
        categories = {match[0]: match[1].strip() for match in category_pattern.findall(text)}

        # Match full rule entries like: '1.1.1 Question\nReason'
        rule_pattern = re.compile(r'(\d+\.\d+\.\d+)\s+(.+?)\n\s*(.+?)(?=\n\d+\.\d+\.\d+|\n\d+\.\d+|\Z)', re.DOTALL)
        rules_by_category: Dict[str, List[str]] = {}

        for number, question, reason in rule_pattern.findall(text):
            category_key = ".".join(number.split('.')[:2])
            category_name = categories.get(category_key, f"Section {category_key}")

            # Cleanup formatting
            cleaned_q = re.sub(r'\s+', ' ', question).strip()
            cleaned_r = re.sub(r'\s+', ' ', reason).strip()

            combined = f"{number} — {cleaned_q}\nWhy: {cleaned_r}"

            if category_name not in rules_by_category:
                rules_by_category[category_name] = []
            rules_by_category[category_name].append(combined)

        return rules_by_category

    def get_rules_as_documents(self) -> List[Document]:
        """Returns structured rules as LangChain Document objects."""
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
        """Returns all checklist rules formatted as markdown."""
        rules_by_category = self.extract_rules()
        markdown = ""

        for category, rules in rules_by_category.items():
            markdown += f"### {category}\n"
            for rule in rules:
                markdown += f"- {rule}\n"
            markdown += "\n"

        return markdown
