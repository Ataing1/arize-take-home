# Project summary

For my take home rag project. i decided to build a rag that was capable of answering questions related to the research paper "Attention Is All You Need" as well as research papers that were linked inside "Attention Is All You Need". I chose to do this because Arize is an AI evaluation company, so i felt it would be on theme to build something that would help analyze or understand AI systems themselves.

I used langchain to build the rag since it has extensive support and is easy to use. I chose to use open ai's embeddings model arbitrarily since it's unrelated to which LLM I use. The most important thing was that i needed to use the same embedding model for embedding all of the documents and all of the queries provided by the LLM. For the model I chose claude 3.5 sonnet because i use it the most on a daily basis and it adheres much more strictly to the prompt. I had some issues with gpt 4 giving answers with more than the context of the retrievals which i talk about more in the things i did poorly section

for the data collection, i wrote a script to start with the seed document "Attention Is All You Need" and then it will download the document, fetch all metadata and then, find all links to other arxiv ids and scrape all of those related research papers. 

# Setup guide

## installation and setup

using python 3.12.8

`python -m venv venv` to make the new python virtual environment

`source venv/bin/activate` to activate the virual environemnt

`pip freeze` to check that it's a fresh virtual environment (should be empty)

`pip install -r requirements.txt` to download the requirements for the project

## fetching the data

`python paper_collector.py` to download the pdf for the seed article and recursively articles mentioned in the main pdf. 

`python rag.py` to run the actual q and a system

## TODO nice cli to see the options of what you can do. 
- view documents
- add documents (by arxiv id)

# Good questions to ask that relate to the documents stored:

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


# lessons and things that went well and didn't go well

So for most of the time i spent implementing this project i used open ai GPT 4o. in this case gpt 4 was too smart for it's own good, as it was able to answer the questions with or without the added context of what was retrieved from the vector store. gpt4 was bad because it failed to adhere to the prompt where i told it to only use the context from the retrievals to answer the questions. When i switched to claude, claude began to adhere strictly to the context and highlighted some problem with my rag system. the data collection was good, but the questions i was testing the RAG system on where bad. i was asking the RAG questions tangentially related like "How did different research teams adapt the Transformer architecture for their specific tasks?" when that's not something that can be retrieved from "Attention is all you need" and those papers that it referenced. If i had more time i would try to include papers which cited "Attention is all you need" which whose context might have been able to answer the question "How did different research teams adapt the Transformer architecture for their specific tasks?"

    

