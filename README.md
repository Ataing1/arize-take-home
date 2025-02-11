# Project Summary

For my take home RAG project, I decided to build a system capable of answering questions related to the research paper "Attention Is All You Need" as well as research papers cited within it. I chose this approach because Arize is an AI evaluation company, so I felt it would be thematic to build something that helps analyze and understand AI systems themselves.

I used LangChain to build the RAG system due to its extensive support and ease of use. I chose to use OpenAI's embeddings model since it's independent of which LLM I use. The most important consideration was using the same embedding model for both document embeddings and LLM query embeddings. For the model, I chose Claude 3.5 Sonnet because I use it frequently and it adheres more strictly to prompts. I encountered some issues with GPT-4 providing answers beyond the context of the retrievals, which I discuss further in the "Lessons Learned" section.

For data collection, I wrote a script that starts with the seed document "Attention Is All You Need," downloads it, fetches all metadata, and then finds and scrapes all related research papers referenced through arXiv IDs. 

I added a feature where you can enter your own seed article to create a dataset around an arxiv article that interests you! 

# Setup guide

## installation and setup

using python 3.12.8

`python -m venv venv` to make the new python virtual environment

`source venv/bin/activate` to activate the virual environemnt

`pip freeze` to check that it's a fresh virtual environment (should be empty)

`pip install -r requirements.txt` to download the requirements for the project

```
## .env:
 you will need an anthropic api key. rename the file .env.example to .env and paste in your real key. 

 should start like sk-ant-api**-...

```

## fetching the data

1. `python paper_collector.py` to download the pdf for the seed article and recursively articles mentioned in the main pdf. 

Arguments:
- `--seed-paper`: ArXiv ID in format YYMM.NNNNN (default: 1706.03762 - "Attention is All You Need")
- `--max-depth`: Maximum depth for recursive paper collection (default: 2)
- `--max-papers`: Maximum number of papers to collect (default: 50)

Example usage:
```bash
# Use defaults (Attention is All You Need paper)
python paper_collector.py

# Collect a different paper with custom settings
python paper_collector.py --seed-paper 1602.02410 --max-depth 3 --max-papers 100
```

2. `python rag.py` to run the actual q and a system
    use -- verbose to see which documents got retrieved

# Good questions to ask that relate to the documents stored (these questions target the default paper "Attention is All you need):

## 1. What were the key problems or limitations in sequence modeling that the transformer architecture aimed to address?

According to the paper "Attention Is All You Need," the Transformer architecture aimed to address two main limitations of existing sequence models:

1. Sequential computation constraint: Recurrent models were inherently sequential in nature, processing inputs one position at a time, which limited parallelization and became problematic with longer sequences due to memory constraints.

2. Learning long-range dependencies: Previous architectures like ConvS2S and ByteNet required more operations to relate signals between distant positions (growing linearly or logarithmically with distance), making it difficult to learn long-range dependencies between positions in the input and output sequences.

The Transformer addressed these issues by relying entirely on self-attention mechanisms, which allowed for parallel computation and required only a constant number of operations to relate any two positions in a sequence.

## 2. How does the transformer's self-attention mechanism differ from attention mechanisms in papers it cites?

According to the paper "Attention Is All You Need," the Transformer is the first transduction model to rely entirely on self-attention without using recurrent networks or convolutions. While earlier papers used attention mechanisms, they typically used them in conjunction with recurrent networks. The key difference is that the Transformer uses self-attention as its primary mechanism for modeling dependencies between input and output sequences, rather than as a supplementary component to RNNs, allowing for significantly more parallelization than recurrent architectures.

## 3. What computational efficiency challenges did earlier sequence models face, and how did the transformer paper propose to solve them?

According to the paper, earlier recurrent models faced two main computational efficiency challenges: 1) They had inherently sequential computation that precluded parallelization within training examples, which became particularly problematic with longer sequences due to memory constraints, and 2) They required O(n) sequential operations for processing sequences of length n. The Transformer solved these challenges by replacing recurrence with self-attention, which allows for constant O(1) sequential operations and significantly more parallelization, though this comes at the cost of O(n²·d) complexity per layer which the authors note is actually faster than recurrent layers when sequence length is smaller than representation dimensionality.

## 4. What were the theoretical foundations or mathematical concepts that the transformer architecture built upon?

Based on the provided context, the Transformer architecture primarily built upon attention mechanisms and self-attention concepts that were previously used in sequence modeling tasks. The paper "Attention Is All You Need" introduced the first sequence transduction model that relied entirely on self-attention to compute representations of input and output, moving away from traditional recurrent neural networks (RNNs) and convolutions. An attention function was mathematically described as mapping a query and a set of key-value pairs to an output, where these components are all vectors, and the output is computed as a weighted sum.

## 5. How did the authors validate their architecture choices in the original transformer paper? What were their key experiments?

Based on the provided context, the authors validated their architecture through several key experiments:

1. Machine Translation: They tested the Transformer on WMT 2014 English-to-German and English-to-French translation tasks, where the big Transformer model achieved state-of-the-art results - 28.4 BLEU on English-to-German and 41.0 BLEU on English-to-French, surpassing previous models at a fraction of the training cost.

2. Model Variations: They conducted ablation studies by varying different components of the base model (including number of attention heads, attention key size, model size, and dropout rates) and measured performance changes on English-to-German translation.

3. English Constituency Parsing: To test generalization, they evaluated a 4-layer Transformer on parsing tasks using both limited data (WSJ dataset with 40K sentences) and semi-supervised settings (17M sentences), demonstrating strong performance even without task-specific tuning.

## 6. What were the key baseline models that the original transformer paper compared against?

Based on the provided context, the original transformer paper compared against several key baseline models in machine translation tasks, specifically on the WMT 2014 English-to-German and English-to-French tests:

1. ByteNet
2. Deep-Att + PosUnk (both single model and ensemble)
3. GNMT + RL (both single model and ensemble)
4. ConvS2S (both single model and ensemble)
5. MoE (Mixture of Experts)

The Transformer outperformed all these baselines, with the big Transformer model achieving 28.4 BLEU on English-to-German (more than 2.0 BLEU points better than previous best) and 41.8 BLEU on English-to-French. Notably, these results were achieved at a fraction of the training cost compared to the baseline models, with training taking just 3.5 days on 8 P100 GPUs.

## 7. What were the primary training challenges discussed in the transformer paper and how did they address them?

Based on the provided context, the paper discussed several key training aspects and challenges:

1. Data and Batching:
- Used WMT 2014 datasets (4.5M sentence pairs for English-German, 36M for English-French)
- Used byte-pair encoding for vocabulary
- Batched sentences by approximate sequence length (25,000 source/target tokens per batch)

2. Hardware and Training Schedule:
- Trained on 8 NVIDIA P100 GPUs
- Base models took 0.4 seconds per step (100,000 steps/12 hours total)
- Big models took 1.0 seconds per step (300,000 steps/3.5 days total)

3. Optimization and Regularization:
- Used Adam optimizer with custom learning rate schedule
- Employed three types of regularization
- Applied residual dropout (rate of 0.1)
- Used label smoothing (value of 0.1) which hurt perplexity but improved accuracy and BLEU scores

The paper focused more on addressing technical training considerations rather than specific challenges, with emphasis on optimization strategies and regularization techniques to improve model performance.

    


# Lessons Learned
During the implementation of this project, I experienced several key insights:

1. Model Selection:
    - Initially used OpenAI GPT-4o-mini, which proved too capable as it answered questions with or without the retrieved context. It was also bad because it didn't adhere to the prompt which asked it to strictly use the content gathered from the retrieval step. 
    - moved on to Claude, which strictly adhered to the context and highlighted issues in the RAG system
    - improving the types of questions asked gave claude high prompt adherence and also showed the accuracy of the data retrieval


2. Data Collection:
    - The document collection process worked well
     Question testing revealed limitations in the document scope
    -  Questions like "How did different research teams adapt the Transformer architecture for their specific tasks?" couldn't be answered from the original paper and its references alone
    - Future improvement: Include papers that cite "Attention Is All You Need" to provide broader context

3. System Performance:
    - After refining questions to focus on the actual content of stored PDFs, Claude performed well and maintained strict adherence to the prompt

# Future additions
if i had more time on the project i might add a voice interface. i'd like to be able to speak to the computer and have it read back out the answer to me. 

i'd also be interested in figuring out ways to find articles that reference the seed article. this way you can ask questions about multiple related research projects and how they differ from one another. 