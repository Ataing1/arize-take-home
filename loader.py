from pathlib import Path
import json
from typing import List, Dict
import re

from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


class ResearchLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.papers_dir = self.data_dir / "papers"
        self.metadata_path = self.data_dir / "metadata.json"

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Improved text splitter with larger chunks and better overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=400,
            length_function=len,
            separators=["\n## ", "\n\n", "\n", " ", ""], 
        )

    def _flatten_metadata(self, paper_info: Dict) -> Dict:
        """Enhanced metadata flattening with more context"""
        return {
            "source_type": "paper",
            "title": paper_info["title"],
            "year": str(paper_info["year"]),
            "filename": paper_info["filename"],
            "authors": ", ".join(paper_info.get("authors", [])),  # Add authors
            "references": ", ".join(paper_info.get("references", [])),  # Add references
            "arxiv_id": paper_info.get("arxiv_id", ""),  # Add arXiv ID
            "depth": str(paper_info.get("depth", 0)),  # Add depth information
        }

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key technical concepts from text"""
        # Simple keyword-based extraction (can be enhanced with NLP)
        technical_patterns = [
            r'\b(?:neural|deep learning|transformer|attention|model|architecture)\w*\b',
            r'\b(?:algorithm|method|technique|approach)\w*\b',
            r'\b(?:training|learning|optimization)\w*\b'
        ]
        concepts = set()
        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.update(matches)
        return list(concepts)

    def _identify_section(self, text: str) -> str:
        """Identify the section of the paper"""
        section_markers = {
            'abstract': ['abstract', 'summary'],
            'introduction': ['introduction', 'background', '1.'],
            'methodology': ['method', 'approach', 'model', '3.'],
            'results': ['results', 'evaluation', 'experiments', '4.'],
            'conclusion': ['conclusion', 'discussion', 'future work'],
            'references': ['references', 'bibliography']
        }
        
        text_lower = text.lower()
        for section, markers in section_markers.items():
            if any(marker in text_lower for marker in markers):
                return section
        return 'body'

    def _enhance_metadata(self, doc: Document, paper_info: Dict) -> Document:
        """Enhanced metadata enrichment"""
        # Basic metadata
        flat_metadata = self._flatten_metadata(paper_info)
        
        # Enhanced metadata
        enhanced_metadata = {
            **flat_metadata,
            'key_concepts': self._extract_key_concepts(doc.page_content),
            'section': self._identify_section(doc.page_content),
            'content_length': len(doc.page_content),
            'has_equations': bool(re.search(r'\$.*?\$', doc.page_content)),
            'has_citations': bool(re.search(r'\[\d+\]|\(\w+ et al\., \d{4}\)', doc.page_content))
        }
        
        doc.metadata.update(enhanced_metadata)
        return doc

    def load_papers(self) -> List[Document]:
        """Enhanced paper loading with improved chunking and metadata"""
        papers = []

        for paper_info in self.metadata["papers"]:
            pdf_path = self.papers_dir / f"{paper_info['filename']}.pdf"
            if pdf_path.exists():
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                
                # Process each page to identify major sections
                processed_pages = []
                current_section = ""
                for page in pages:
                    section = self._identify_section(page.page_content)
                    if section != "body":
                        current_section = section
                    page.metadata["section"] = current_section
                    processed_pages.append(page)

                # Improved chunking with section awareness
                chunks = self.text_splitter.split_documents(processed_pages)
                
                # Enhance metadata for each chunk
                enhanced_chunks = [self._enhance_metadata(chunk, paper_info) for chunk in chunks]
                papers.extend(enhanced_chunks)

        return papers

    def load_all(self) -> List[Document]:
        """Load all documents and return combined list"""
        papers = self.load_papers()
        return papers

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create a vector store from the documents"""
        embeddings = OpenAIEmbeddings()

        # Create Chroma vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(self.data_dir / "vector_store"),
            collection_metadata={
                "hnsw:space": "cosine"
            },  # Explicitly set distance metric
        )

        return vector_store


# def main():
#     """Example usage"""
#     # Initialize loader
#     loader = ResearchLoader()

#     # Load all documents
#     print("Loading documents...")
#     documents = loader.load_all()
#     print(f"Loaded {len(documents)} document chunks")

#     # Create vector store
#     print("Creating vector store...")
#     vector_store = loader.create_vector_store(documents)
#     print("Vector store created successfully")

#     # Example metadata
#     for i, doc in enumerate(documents[:2]):
#         print(f"\nDocument {i + 1} metadata:")
#         print(json.dumps(doc.metadata, indent=2))


# if __name__ == "__main__":
#     main()
