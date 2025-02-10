from pathlib import Path
import json
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


class ResearchLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.papers_dir = self.data_dir / "papers"
        self.concepts_dir = self.data_dir / "concepts"
        self.metadata_path = self.data_dir / "metadata.json"
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def _flatten_metadata(self, paper_info: Dict) -> Dict:
        """Convert complex metadata into Chroma-compatible flat structure"""
        # Join lists into strings with a delimiter
        key_concepts = "|".join(paper_info["key_concepts"])
        
        return {
            "source_type": "paper",
            "title": paper_info["title"],
            "year": str(paper_info["year"]),  # Convert to string for compatibility
            "key_concepts": key_concepts,
            "filename": paper_info["filename"]
        }

    def load_papers(self) -> List[Document]:
        """Load and split PDF papers"""
        papers = []
        
        for paper_info in self.metadata["papers"]:
            pdf_path = self.papers_dir / f"{paper_info['filename']}.pdf"
            if pdf_path.exists():
                # Load PDF
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                
                # Split into chunks
                chunks = self.text_splitter.split_documents(pages)
                
                # Add flattened metadata to each chunk
                flat_metadata = self._flatten_metadata(paper_info)
                for chunk in chunks:
                    chunk.metadata.update(flat_metadata)
                
                papers.extend(chunks)
        
        return papers

    def load_concepts(self) -> List[Document]:
        """Load and split markdown concept files"""
        concepts = []
        
        for concept in self.metadata["concepts"]:
            concept_path = self.concepts_dir / f"{concept}.md"
            if concept_path.exists():
                # Load markdown
                loader = UnstructuredMarkdownLoader(str(concept_path))
                concept_doc = loader.load()
                
                # Split into chunks
                chunks = self.text_splitter.split_documents(concept_doc)
                
                # Add metadata to each chunk
                for chunk in chunks:
                    chunk.metadata.update({
                        "source_type": "concept",
                        "concept_name": concept,
                        "year": "N/A",  # Added for consistency
                        "title": f"{concept.capitalize()} Concept"  # Added for consistency
                    })
                
                concepts.extend(chunks)
        
        return concepts

    def load_all(self) -> List[Document]:
        """Load all documents and return combined list"""
        papers = self.load_papers()
        concepts = self.load_concepts()
        return papers + concepts

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create a vector store from the documents"""
        embeddings = OpenAIEmbeddings()
        
        # Create Chroma vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(self.data_dir / "vector_store"),
            collection_metadata={"hnsw:space": "cosine"}  # Explicitly set distance metric
        )
        
        return vector_store

def main():
    """Example usage"""
    # Initialize loader
    loader = ResearchLoader()
    
    # Load all documents
    print("Loading documents...")
    documents = loader.load_all()
    print(f"Loaded {len(documents)} document chunks")
    
    # Create vector store
    print("Creating vector store...")
    vector_store = loader.create_vector_store(documents)
    print("Vector store created successfully")
    
    # Example metadata
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1} metadata:")
        print(json.dumps(doc.metadata, indent=2))

if __name__ == "__main__":
    main()