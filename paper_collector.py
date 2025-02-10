import PyPDF2
import re
import requests
import time
import xml.etree.ElementTree as ET
from collections import deque
from typing import Set, Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

class RecursivePaperCollector:
    def __init__(self, base_dir: str = "data", max_depth: int = 3, max_papers: int = 200):
        self.base_dir = Path(base_dir)
        self.papers_dir = self.base_dir / "papers"
        self.metadata_file = self.base_dir / "metadata.json"
        self.max_depth = max_depth
        self.max_papers = max_papers
        self.papers_processed = set()
        
        # Create directories if they don't exist
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
        print(self.metadata)

    def _load_metadata(self) -> Dict:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"papers": [], "citations": {}, "last_updated": None}

    async def fetch_paper_metadata(self, arxiv_id: str) -> Optional[Tuple[str, List[str], int]]:
        """Fetch paper title, authors, and year from arXiv API"""
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Parse XML response
                namespace = {'atom': 'http://www.w3.org/2005/Atom'}
                root = ET.fromstring(response.content)
                
                # Extract title, authors, and published date
                title_elem = root.find('.//atom:entry/atom:title', namespace)
                author_elems = root.findall('.//atom:entry/atom:author/atom:name', namespace)
                published_elem = root.find('.//atom:entry/atom:published', namespace)
                
                if title_elem is not None and published_elem is not None:
                    title = title_elem.text.strip()
                    authors = [author.text.strip() for author in author_elems]
                    # Extract year from published date (format: YYYY-MM-DDT...)
                    year = int(published_elem.text[:4])
                    return title, authors, year
                    
            time.sleep(1)  # Be nice to arXiv API
        except Exception as e:
            print(f"Error fetching metadata for {arxiv_id}: {e}")
        return None

    def extract_arxiv_ids(self, text: str) -> Set[str]:
        """Extract arXiv IDs from text using regex patterns"""
        # Pattern for new arXiv IDs (e.g., 2103.00020)
        pattern1 = r'(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)'
        # Pattern for old arXiv IDs (e.g., math.GT/0309136)
        pattern2 = r'(?:arXiv:)?([a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)'
        
        ids = set()
        ids.update(re.findall(pattern1, text, re.IGNORECASE))
        ids.update(re.findall(pattern2, text, re.IGNORECASE))
        return ids

    def extract_references_from_pdf(self, pdf_path: Path) -> Set[str]:
        """Extract potential arXiv IDs from PDF references"""
        arxiv_ids = set()
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Focus on last few pages (usually where references are)
                start_page = max(0, len(pdf_reader.pages) - 5)
                for page_num in range(start_page, len(pdf_reader.pages)):
                    text = pdf_reader.pages[page_num].extract_text()
                    arxiv_ids.update(self.extract_arxiv_ids(text))
                    
        except Exception as e:
            print(f"Error extracting references from {pdf_path}: {e}")
        
        return arxiv_ids

    async def download_paper(self, arxiv_id: str) -> Optional[Path]:
        """Download paper from arXiv"""
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        filename = f"paper_{arxiv_id.replace('/', '_')}.pdf"
        output_path = self.papers_dir / filename
        
        if output_path.exists():
            return output_path
            
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                time.sleep(1)  # Be nice to arXiv servers
                return output_path
        except Exception as e:
            print(f"Error downloading {arxiv_id}: {e}")
        
        return None

    async def process_paper_recursively(self, arxiv_id: str, depth: int = 0):
        """Recursively process a paper and its references"""
        if depth >= self.max_depth or len(self.papers_processed) >= self.max_papers:
            return
            
        if arxiv_id in self.papers_processed:
            return
            
        print(f"Processing {arxiv_id} at depth {depth}")
        self.papers_processed.add(arxiv_id)
        
        # Fetch metadata first
        metadata_result = await self.fetch_paper_metadata(arxiv_id)
        if not metadata_result:
            print(f"Failed to fetch metadata for {arxiv_id}")
            return
            
        title, authors, year = metadata_result
        
        # Download paper
        pdf_path = await self.download_paper(arxiv_id)
        if not pdf_path:
            return
            
        # Extract references
        reference_ids = self.extract_references_from_pdf(pdf_path)
        
        # Update metadata
        paper_info = {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "year": year,
            "filename": pdf_path.stem,
            "references": list(reference_ids),
            "depth": depth,
            "date_added": datetime.now().isoformat()
        }
        self.metadata["papers"].append(paper_info)
        print(self.metadata)
        self.metadata["citations"][arxiv_id] = list(reference_ids)
        self._save_metadata()
        
        # Process references recursively
        for ref_id in reference_ids:
            await self.process_paper_recursively(ref_id, depth + 1)

    def _save_metadata(self):
        """Save metadata to file"""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)

    def get_citation_graph(self) -> Dict:
        """Return citation graph data"""
        return {
            "nodes": [{"id": paper["arxiv_id"], 
                      "title": paper["title"],
                      "year": paper["year"],
                      "depth": paper["depth"]} 
                     for paper in self.metadata["papers"]],
            "edges": [(paper_id, ref_id) 
                     for paper_id, refs in self.metadata["citations"].items() 
                     for ref_id in refs]
        }

    def get_collection_stats(self) -> Dict:
        """Get statistics about the paper collection"""
        depths = {}
        for paper in self.metadata["papers"]:
            depth = paper.get("depth", 0)
            depths[depth] = depths.get(depth, 0) + 1

        return {
            "total_papers": len(self.metadata["papers"]),
            "papers_by_depth": depths,
            "total_citations": sum(len(refs) for refs in self.metadata["citations"].values()),
            "last_updated": self.metadata["last_updated"]
        }

# Example usage
async def main():
    collector = RecursivePaperCollector(max_depth=2, max_papers=50)
    
    # Start with a seed paper (Attention is All You Need)
    seed_paper = "1706.03762"
    await collector.process_paper_recursively(seed_paper)
    
    # Get statistics
    stats = collector.get_collection_stats()
    print(f"Collection stats: {json.dumps(stats, indent=2)}")
    
    # Get citation graph
    graph = collector.get_citation_graph()
    print(f"Citation graph: {json.dumps(graph, indent=2)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())