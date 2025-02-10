i'm thinking i want to build a rag that can teach me about the most notable ai research papers and their findings as well as common model architectures and their evolution like transformers, CNNs, and RNNs. 

i plan to use langchain with a CLI application. i'll probrably end up using alot of jupyter notebook to test various stages of development


using python 3.12.8

`python -m venv venv` to make the new python virtual environment

`source venv/bin/activate` to activate the virual environemnt

`pip freeze` to check that it's a fresh virtual environment (should be empty)

 i generated a dataset starting a seed arxiv paper, and recursively scraping the other arxiv papers that appeared in the current paper. 

 

# TODO - at the end, add a feature allowing users to input a valid arxiv link and it should load and add that research paper to the list.

# TODO nice cli to see the options of what you can do. 
- view documents
- add documents (by arxiv id)


