# Hilti-AI

This is a project I developed during the Hilti IT Competition. Its function is to let the user query for anything related to Hitli and the AI will be able to look through its knowledge base and return the answer and as well as the top 5 source documents that it used as reference for the answer. This solution is an example of RAG being implemented in order to enhance user queries while also returning the source of information that the AI refers to in order to get the answer. In the chunking step, i opted for a Cluster Semantic Chunking + spaCy NLP for splitting text and arranging them into clusters that optimized automatically. For more information regarding chunking approaches, visit https://research.trychroma.com/evaluating-chunking as they provide a detailed report regarding the contemporary approaches to chunking.

I used an open source embedding model (all-mpnet-base-v2) from HuggingFace for simplicity and ease of use and stored the embeddings into a vector DB using Chroma. I also used a reranking model, "rerank-v3.5" by Cohere which allows me to evaluate and pass more relevant chunks of information to the LLM. I utilized google's generative AI model such as gemini-2.0-flash as the LLM, and finally, The user interface is created using gradio.

# Sample Pictures
![sample image](images/Screenshot%202025-05-13%20101702.png)

# Diagram Representation
![sample image](images/Screenshot%202025-05-13%20113040.png)