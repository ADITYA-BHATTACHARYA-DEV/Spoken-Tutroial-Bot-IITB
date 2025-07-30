#!/usr/bin/env python3
import argparse
import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Optional

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent import RAGAgent
from checklist_extractor import ChecklistRulesExtractor

load_dotenv()

class RAGCLI:
    def __init__(self):
        self.agent = RAGAgent()
        
    def print_banner(self):
        print("""
        ██████╗  █████╗  ██████╗ 
        ██╔══██╗██╔══██╗██╔════╝ 
        ██████╔╝███████║██║  ███╗
        ██╔══██╗██╔══██║██║   ██║
        ██║  ██║██║  ██║╚██████╔╝
        ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ 
        Retrieval-Augmented Generation System
        """)

    def process_single_document(self, file_path: str, output_dir: str):
        """Process a single PDF document"""
        try:
            if not os.path.exists(file_path):
                print(f"Error: File not found - {file_path}")
                return

            # Create temporary document list
            doc = self.agent.load_documents_from_list([file_path])
            if not doc:
                print("Failed to load document")
                return

            self.agent.create_document_index(doc)
            
            # Generate output filename
            base_name = os.path.basename(file_path)
            output_file = os.path.join(
                output_dir, 
                f"rewritten_{base_name.replace('.pdf', '.md')}"
            )
            
            # Process document
            rewritten_content = self.agent.generate_rewritten_content(
                query=doc[0].page_content[:500],
                document_content=doc[0].page_content
            )
            
            # Save output
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(rewritten_content)
            
            print(f"Successfully processed: {file_path}")
            print(f"Output saved to: {output_file}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    def run(self):
        self.print_banner()
        
        parser = argparse.ArgumentParser(
            description="RAG Document Rewriting System CLI",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Main arguments
        parser.add_argument(
            "--checklist", 
            help="Path to checklist PDF",
            default=os.getenv("CHECKLIST_PDF")
        )
        parser.add_argument(
            "--docs", 
            help="Path to documents folder or single PDF file",
            default=os.getenv("DOCS_FOLDER")
        )
        parser.add_argument(
            "--output", 
            help="Output directory",
            default=os.getenv("OUTPUT_DIR", "./output")
        )
        parser.add_argument(
            "--list-rules",
            help="Display extracted rules without processing",
            action="store_true"
        )
        
        args = parser.parse_args()
        
        try:
            # Load checklist
            print("\n[1/3] Loading checklist...")
            self.agent.load_checklist(args.checklist)
            
            if args.list_rules:
                # Just display rules and exit
                extractor = ChecklistRulesExtractor(args.checklist)
                print("\nExtracted Rules:")
                print("=" * 50)
                print(extractor.get_formatted_rules())
                return
            
            # Process documents
            if os.path.isfile(args.docs):
                print("\n[2/3] Processing single document...")
                self.process_single_document(args.docs, args.output)
            else:
                print("\n[2/3] Processing documents folder...")
                print(f"Input directory: {args.docs}")
                print(f"Output directory: {args.output}")
                
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("Aborted by user")
                    return
                
                print("\n[3/3] Rewriting documents...")
                self.agent.process_documents(args.output)
            
            print("\nOperation completed successfully")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    cli = RAGCLI()
    cli.run()