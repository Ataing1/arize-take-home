from pathlib import Path
import json
from typing import List, Dict

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
        self.concepts_dir = self.data_dir / "concepts"
        self.metadata_path = self.data_dir / "metadata.json"

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
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

    def load_papers(self) -> List[Document]:
        """Enhanced paper loading with better metadata and chunk handling"""
        papers = []

        for paper_info in self.metadata["papers"]:
            pdf_path = self.papers_dir / f"{paper_info['filename']}.pdf"
            if pdf_path.exists():
                # Load PDF
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()

                # Enhanced text splitting with better parameters
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,  # Smaller chunks for more precise retrieval
                    chunk_overlap=100,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""],  # More granular splitting
                )

                chunks = self.text_splitter.split_documents(pages)

                # Add enhanced metadata to each chunk
                flat_metadata = self._flatten_metadata(paper_info)
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update(flat_metadata)
                    chunk.metadata["chunk_index"] = i  # Add chunk position information

                    # Add section detection
                    if "references" in chunk.page_content.lower():
                        chunk.metadata["section"] = "references"
                    elif "abstract" in chunk.page_content.lower():
                        chunk.metadata["section"] = "abstract"

                papers.extend(chunks)

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
