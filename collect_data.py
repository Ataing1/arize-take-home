import os
import requests
import json
from pathlib import Path
from typing import Dict, List

class PaperCollector:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.papers_dir = self.base_dir / "papers"
        self.concepts_dir = self.base_dir / "concepts"
        self.metadata_file = self.base_dir / "metadata.json"
        
        # Create directories if they don't exist
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.concepts_dir.mkdir(parents=True, exist_ok=True)

    def download_paper(self, arxiv_id: str, filename: str) -> bool:
        """Download a paper from arXiv"""
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(url)
        
        if response.status_code == 200:
            paper_path = self.papers_dir / f"{filename}.pdf"
            with open(paper_path, "wb") as f:
                f.write(response.content)
            return True
        return False

    def create_concept_file(self, concept_name: str, content: str):
        """Create a markdown file for a concept"""
        concept_path = self.concepts_dir / f"{concept_name}.md"
        with open(concept_path, "w", encoding="utf-8") as f:
            f.write(content)

    def save_metadata(self, metadata: Dict):
        """Save metadata about papers and concepts"""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

def main():
    collector = PaperCollector()
    
    # Key papers to download
    papers = [
        {
            "title": "Attention Is All You Need",
            "arxiv_id": "1706.03762",
            "filename": "attention_transformer",
            "year": 2017,
            "key_concepts": ["Transformer architecture", "Self-attention", "Multi-head attention"]
        },
        {
            "title": "Deep Residual Learning for Image Recognition",
            "arxiv_id": "1512.03385",
            "filename": "resnet",
            "year": 2015,
            "key_concepts": ["Residual connections", "Deep networks", "Gradient flow"]
        },
        {
            "title": "Language Models are Few-Shot Learners",
            "arxiv_id": "2005.14165",
            "filename": "gpt3",
            "year": 2020,
            "key_concepts": ["Few-shot learning", "Scale", "Emergent abilities"]
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "arxiv_id": "1810.04805",
            "filename": "bert",
            "year": 2018,
            "key_concepts": ["Bidirectional attention", "Masked language modeling", "Pre-training"]
        }
    ]
    
    # Download papers
    for paper in papers:
        print(f"Downloading {paper['title']}...")
        success = collector.download_paper(paper['arxiv_id'], paper['filename'])
        if success:
            print(f"Successfully downloaded {paper['title']}")
        else:
            print(f"Failed to download {paper['title']}")

    # Create concept files
    concepts = {
        "transformers": """# Transformer Architecture

The Transformer architecture, introduced in the "Attention Is All You Need" paper, is a groundbreaking neural network design that revolutionized natural language processing.

## Key Components:
1. Self-Attention Mechanism
2. Multi-Head Attention
3. Position Encodings
4. Feed-Forward Networks

## Why It's Important
- Handles long-range dependencies better than RNNs
- Enables parallel processing
- Forms the foundation for models like BERT and GPT
""",
        "attention": """# Attention Mechanisms

Attention mechanisms allow neural networks to focus on relevant parts of the input when producing output.

## Types of Attention:
1. Self-Attention
2. Multi-Head Attention
3. Cross-Attention

## Applications:
- Natural Language Processing
- Computer Vision
- Speech Recognition
""",
        "neural_networks": """# Neural Networks Fundamentals

Neural networks are computing systems inspired by biological neural networks in animal brains.

## Key Concepts:
1. Neurons and Layers
2. Activation Functions
3. Backpropagation
4. Gradient Descent

## Evolution:
- Feed-forward Networks
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformers
"""
    }
    
    for concept_name, content in concepts.items():
        print(f"Creating concept file for {concept_name}...")
        collector.create_concept_file(concept_name, content)
    
    # Save metadata
    metadata = {
        "papers": papers,
        "concepts": list(concepts.keys())
    }
    collector.save_metadata(metadata)
    print("Metadata saved successfully")

if __name__ == "__main__":
    main()