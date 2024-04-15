# %%
# Download PDF file
import os
import requests

# Get PDF document
pdf_path = "human-nutrition-text.pdf"

# Download PDF if it doesn't already exist
if not os.path.exists(pdf_path):
  print("File doesn't exist, downloading...")

  # The URL of the PDF you want to download
  url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

  # The local filename to save the downloaded file
  filename = pdf_path

  # Send a GET request to the URL
  response = requests.get(url)

  # Check if the request was successful
  if response.status_code == 200:
      # Open a file in binary write mode and save the content to it
      with open(filename, "wb") as file:
          file.write(response.content)
      print(f"The file has been downloaded and saved as {filename}")
  else:
      print(f"Failed to download the file. Status code: {response.status_code}")
else:
  print(f"File {pdf_path} exists.")

# %%
# Requires !pip install PyMuPDF, see: https://github.com/pymupdf/pymupdf
import fitz # (pymupdf, found this is better than pypdf for our use case, note: licence is AGPL-3.0, keep that in mind if you want to use any code commercially)
from tqdm.auto import tqdm # for progress bars, requires !pip install tqdm

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

    # Other potential text formatting functions can go here
    return cleaned_text

# Open PDF and get lines/pages
# Note: this only focuses on text, rather than images/figures etc
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number - 41,  # adjust page numbers since our PDF starts on page 42
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                "text": text})
    return pages_and_texts

pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
pages_and_texts[:2]

# %% [markdown]
# Now let's get a random sample of the pages.

# %%
import random

random.sample(pages_and_texts, k=3)

# %%
import pandas as pd

df = pd.DataFrame(pages_and_texts)
df.head()

# %%
# Get stats
df.describe().round(2)

# %% [markdown]
# Okay, looks like our average token count per page is 287.
# 
# For this particular use case, it means we could embed an average whole page with the `all-mpnet-base-v2` model (this model has an input capacity of 384).

# %% [markdown]
# ### Further text processing (splitting pages into sentences)
# 
# The ideal way of processing text before embedding it is still an active area of research.
# 
# A simple method I've found helpful is to break the text into chunks of sentences.
# 
# As in, chunk a page of text into groups of 5, 7, 10 or more sentences (these values are not set in stone and can be explored).
# 
# But we want to follow the workflow of:
# 
# `Ingest text -> split it into groups/chunks -> embed the groups/chunks -> use the embeddings`
# 
# Some options for splitting text into sentences:
# 
# 1. Split into sentences with simple rules (e.g. split on ". " with `text = text.split(". ")`, like we did above).
# 2. Split into sentences with a natural language processing (NLP) library such as [spaCy](https://spacy.io/) or [nltk](https://www.nltk.org/).
# 
# Why split into sentences?
# 
# * Easier to handle than larger pages of text (especially if pages are densely filled with text).
# * Can get specific and find out which group of sentences were used to help within a RAG pipeline.
# 
# > **Resource:** See [spaCy install instructions](https://spacy.io/usage).
# 
# Let's use spaCy to break our text into sentences since it's likely a bit more robust than just using `text.split(". ")`.

# %%
from spacy.lang.en import English # see https://spacy.io/usage for install instructions

nlp = English()

# Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/
nlp.add_pipe("sentencizer")

# Create a document instance as an example
doc = nlp("This is a sentence. This another sentence.")
assert len(list(doc.sents)) == 2

# Access the sentences of the document
list(doc.sents)

# %% [markdown]
# We don't necessarily need to use spaCy, however, it's an open-source library designed to do NLP tasks like this at scale.
# 
# So let's run our small sentencizing pipeline on our pages of text.

# %%
for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)

    # Make sure all sentences are strings
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]

    # Count the sentences
    item["page_sentence_count_spacy"] = len(item["sentences"])

# %%
# Inspect an example
random.sample(pages_and_texts, k=1)

# %% [markdown]
# Wonderful!
# 
# Now let's turn out list of dictionaries into a DataFrame and get some stats.

# %%
df = pd.DataFrame(pages_and_texts)
df.describe().round(2)

# %% [markdown]
# For our set of text, it looks like our raw sentence count (e.g. splitting on `". "`) is quite close to what spaCy came up with.
# 
# Now we've got our text split into sentences, how about we gorup those sentences?

# %% [markdown]
# ### Chunking our sentences together
# 
# Let's take a step to break down our list of sentences/text into smaller chunks.
# 
# As you might've guessed, this process is referred to as **chunking**.
# 
# Why do we do this?
# 
# 1. Easier to manage similar sized chunks of text.
# 2. Don't overload the embedding models capacity for tokens (e.g. if an embedding model has a capacity of 384 tokens, there could be information loss if you try to embed a sequence of 400+ tokens).
# 3. Our LLM context window (the amount of tokens an LLM can take in) may be limited and requires compute power so we want to make sure we're using it as well as possible.
# 
# Something to note is that there are many different ways emerging for creating chunks of information/text.
# 
# For now, we're going to keep it simple and break our pages of sentences into groups of 10 (this number is arbitrary and can be changed, I just picked it because it seemed to line up well with our embedding model capacity of 384).
# 
# On average each of our pages has 10 sentences.
# 
# And an average total of 287 tokens per page.
# 
# So our groups of 10 sentences will also be ~287 tokens long.
# 
# This gives us plenty of room for the text to embedded by our `all-mpnet-base-v2` model (it has a capacity of 384 tokens).
# 
# To split our groups of sentences into chunks of 10 or less, let's create a function which accepts a list as input and recursively breaks into down into sublists of a specified size.

# %%
# Define split size to turn groups of sentences into chunks
num_sentence_chunk_size = 10

# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

# Loop through pages and texts and split sentences into chunks
for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                         slice_size=num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])

# %%
# Sample an example from the group (note: many samples have only 1 chunk as they have <=10 sentences total)
random.sample(pages_and_texts, k=1)

# %%
# Create a DataFrame to get stats
df = pd.DataFrame(pages_and_texts)
df.describe().round(2)

# %% [markdown]
# Note how the average number of chunks is around 1.5, this is expected since many of our pages only contain an average of 10 sentences.

# %% [markdown]
# ### Splitting each chunk into its own item
# 
# We'd like to embed each chunk of sentences into its own numerical representation.
# 
# So to keep things clean, let's create a new list of dictionaries each containing a single chunk of sentences with relative information such as page number as well statistics about each chunk.

# %%
import re

# Split each chunk into its own item
pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo
        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Get stats about the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

        pages_and_chunks.append(chunk_dict)

# How many chunks do we have?
len(pages_and_chunks)

# %%
# View a random sample
random.sample(pages_and_chunks, k=1)

# %% [markdown]
# Excellent!
# 
# Now we've broken our whole textbook into chunks of 10 sentences or less as well as the page number they came from.
# 
# This means we could reference a chunk of text and know its source.
# 
# Let's get some stats about our chunks.

# %%
# Get stats about our chunks
df = pd.DataFrame(pages_and_chunks)
df.describe().round(2)

# %% [markdown]
# Hmm looks like some of our chunks have quite a low token count.
# 
# How about we check for samples with less than 30 tokens (about the length of a sentence) and see if they are worth keeping?

# %%
# Show random chunks with under 30 tokens in length
min_token_length = 30
for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
    print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')

# %% [markdown]
# Looks like many of these are headers and footers of different pages.
# 
# They don't seem to offer too much information.
# 
# Let's filter our DataFrame/list of dictionaries to only include chunks with over 30 tokens in length.

# %%
pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
pages_and_chunks_over_min_token_len[:2]

# %% [markdown]
# Smaller chunks filtered!
# 
# Time to embed our chunks of text!

# %% [markdown]
# ### Embedding our text chunks
# 
# While humans understand text, machines understand numbers best.
# 
# An [embedding](https://vickiboykis.com/what_are_embeddings/index.html) is a broad concept.
# 
# But one of my favourite and simple definitions is "a useful numerical representation".
# 
# The most powerful thing about modern embeddings is that they are *learned* representations.
# 
# Meaning rather than directly mapping words/tokens/characters to numbers directly (e.g. `{"a": 0, "b": 1, "c": 3...}`), the numerical representation of tokens is learned by going through large corpuses of text and figuring out how different tokens relate to each other.
# 
# Ideally, embeddings of text will mean that similar meaning texts have similar numerical representation.
# 
# > **Note:** Most modern NLP models deal with "tokens" which can be considered as multiple different sizes and combinations of words and characters rather than always whole words or single characters. For example, the string `"hello world!"` gets mapped to the token values `{15339: b'hello', 1917: b' world', 0: b'!'}` using [Byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) (or BPE via OpenAI's [`tiktoken`](https://github.com/openai/tiktoken) library). Google has a tokenization library called [SentencePiece](https://github.com/google/sentencepiece).
# 
# Our goal is to turn each of our chunks into a numerical representation (an embedding vector, where a vector is a sequence of numbers arranged in order).
# 
# Once our text samples are in embedding vectors, us humans will no longer be able to understand them.
# 
# However, we don't need to.
# 
# The embedding vectors are for our computers to understand.
# 
# We'll use our computers to find patterns in the embeddings and then we can use their text mappings to further our understanding.
# 
# Enough talking, how about we import a text embedding model and see what an embedding looks like.
# 
# To do so, we'll use the [`sentence-transformers`](https://www.sbert.net/docs/installation.html) library which contains many pre-trained embedding models.
# 
# Specifically, we'll get the `all-mpnet-base-v2` model (you can see the model's intended use on the [Hugging Face model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2#intended-uses)).

# %%
# Requires !pip install sentence-transformers
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device="cpu") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)

# Create a list of sentences to turn into numbers
sentences = [
    "The Sentences Transformers library provides an easy and open-source way to create embeddings.",
    "Sentences can be embedded one by one or as a list of strings.",
    "Embeddings are one of the most powerful concepts in machine learning!",
    "Learn to use embeddings well and you'll be well on your way to being an AI engineer."
]

# Sentences are encoded/embedded by calling model.encode()
embeddings = embedding_model.encode(sentences)
embeddings_dict = dict(zip(sentences, embeddings))

# See the embeddings
for sentence, embedding in embeddings_dict.items():
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

# %% [markdown]
# Woah! That's a lot of numbers.
# 
# How about we do just once sentence?

# %%
single_sentence = "Yo! How cool are embeddings?"
single_embedding = embedding_model.encode(single_sentence)
print(f"Sentence: {single_sentence}")
print(f"Embedding:\n{single_embedding}")
print(f"Embedding size: {single_embedding.shape}")

# %% [markdown]
# Nice! We've now got a way to numerically represent each of our chunks.
# 
# Our embedding has a shape of `(768,)` meaning it's a vector of 768 numbers which represent our text in high-dimensional space, too many for a human to comprehend but machines love high-dimensional space.
# 
# > **Note:** No matter the size of the text input to our `all-mpnet-base-v2` model, it will be turned into an embedding size of `(768,)`. This value is fixed. So whether a sentence is 1 token long or 1000 tokens long, it will be truncated/padded with zeros to size 384 and then turned into an embedding vector of size `(768,)`. Of course, other embedding models may have different input/output shapes.
# 
# How about we add an embedding field to each of our chunk items?
# 
# Let's start by trying to create embeddings on the CPU, we'll time it with the `%%time` magic to see how long it takes.

# %%
%%time

# Uncomment to see how long it takes to create embeddings on CPU
# # Make sure the model is on the CPU
# embedding_model.to("cpu")

# # Embed each chunk one by one
# for item in tqdm(pages_and_chunks_over_min_token_len):
#     item["embedding"] = embedding_model.encode(item["sentence_chunk"])

# %% [markdown]
# Ok not too bad... but this would take a *really* long time if we had a larger dataset.
# 
# Now let's see how long it takes to create the embeddings with a GPU.

# %%
%%time

# Send the model to the GPU
embedding_model.to("cuda") # requires a GPU installed, for reference on my local machine, I'm using a NVIDIA RTX 4090

# Create embeddings one by one on the GPU
for item in tqdm(pages_and_chunks_over_min_token_len):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])

# %% [markdown]
# Woah! Looks like the embeddings get created much faster (~10x faster on my machine) on the GPU!
# 
# You'll likely notice this trend with many of your deep learning workflows. If you have access to a GPU, especially a NVIDIA GPU, you should use one if you can.
# 
# But what if I told you we could go faster again?
# 
# You see many modern models can handle batched predictions.
# 
# This means computing on multiple samples at once.
# 
# Those are the types of operations where a GPU flourishes!
# 
# We can perform batched operations by turning our target text samples into a single list and then passing that list to our embedding model.

# %%
# Turn text chunks into a single list
text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]

# %%
%%time

# Embed all texts in batches
text_chunk_embeddings = embedding_model.encode(text_chunks,
                                               batch_size=32, # you can use different batch sizes here for speed/performance, I found 32 works well for this use case
                                               convert_to_tensor=True) # optional to return embeddings as tensor instead of array

text_chunk_embeddings

# %% [markdown]
# That's what I'm talking about!
# 
# A ~4x improvement (on my GPU) in speed thanks to batched operations.
# 
# So the tip here is to use a GPU when you can and use batched operations if you can too.
# 
# Now let's save our chunks and their embeddings so we could import them later if we wanted.

# %% [markdown]
# ### Save embeddings to file
# 
# Since creating embeddings can be a timely process (not so much for our case but it can be for more larger datasets), let's turn our `pages_and_chunks_over_min_token_len` list of dictionaries into a DataFrame and save it.

# %%
# Save embeddings to file
text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

# %% [markdown]
# And we can make sure it imports nicely by loading it.

# %%
# Import saved file and view
text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)
text_chunks_and_embedding_df_load.head()

# %% [markdown]
# ### Chunking and embedding questions
# 
# > **Which embedding model should I use?**
# 
# This depends on many factors. My best advice is to experiment, experiment, experiment!
# 
# If you want the model to run locally, you'll have to make sure it's feasible to run on your own hardware.
# 
# A good place to see how different models perform on a wide range of embedding tasks is the [Hugging Face Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
# 
# > **What other forms of text chunking/splitting are there?**
# 
# There are a fair few options here too. We've kept it simple with groups of sentences.
# 
# For more, [Pinecone has a great guide on different kinds of chunking](https://www.pinecone.io/learn/chunking-strategies/) including for different kinds of data such as markdown and LaTeX.
# 
# Libraries such as [LangChain also have a good amount of in-built text splitting options](https://python.langchain.com/docs/modules/data_connection/document_transformers/).
# 
# > **What should I think about when creating my embeddings?**
# 
# Our model turns text inputs up to 384 tokens long in embedding vectors of size 768.
# 
# Generally, the larger the vector size, the more information that gets encoded into the embedding (however, this is not always the case, as smaller, better models can outperform larger ones).
# 
# Though with larger vector sizes comes larger storage and compute requirements.
# 
# Our model is also relatively small (420MB) in size compared to larger models that are available.
# 
# Larger models may result in better performance but will also require more compute.
# 
# So some things to think about:
# * Size of input - If you need to embed longer sequences, choose a model with a larger input capacity.
# * Size of embedding vector - Larger is generally a better representation but requires more compute/storage.
# * Size of model - Larger models generally result in better embeddings but require more compute power/time to run.
# * Open or closed - Open models allow you to run them on your own hardware whereas closed models can be easier to setup but require an API call to get embeddings.
# 
# > **Where should I store my embeddings?**
# 
# If you've got a relatively small dataset, for example, under 100,000 examples (this number is rough and only based on first hand experience), `np.array` or `torch.tensor` can work just fine as your dataset.
# 
# But if you've got a production system and want to work with 100,000+ embeddings, you may want to look into a [vector database]( https://en.wikipedia.org/wiki/Vector_database) (these have become very popular lately and there are many offerings).
# 
# ### Document Ingestion and Embedding Creation Extensions
# 
# One major extension to the workflow above would to functionize it.
# 
# Or turn it into a script.
# 
# As in, take all the functionality we've created and package it into a single process (e.g. go from document -> embeddings file).
# 
# So you could input a document on one end and have embeddings come out the other end. The hardest part of this is knowing what kind of preprocessing your text may need before it's turned into embeddings. Cleaner text generally means better results.
# 
# 

# %% [markdown]
# ## 2. RAG - Search and Answer
# 
# We discussed RAG briefly in the beginning but let's quickly recap.
# 
# RAG stands for Retrieval Augmented Generation.
# 
# Which is another way of saying "given a query, search for relevant resources and answer based on those resources".
# 
# Let's breakdown each step:
# * **Retrieval** - Get relevant resources given a query. For example, if the query is "what are the macronutrients?" the ideal results will contain information about protein, carbohydrates and fats (and possibly alcohol) rather than information about which tractors are the best for farming (though that is also cool information).
# * **Augmentation** - LLMs are capable of generating text given a prompt. However, this generated text is designed to *look* right. And it often has some correct information, however, they are prone to hallucination (generating a result that *looks* like legit text but is factually wrong). In augmentation, we pass relevant information into the prompt and get an LLM to use that relevant information as the basis of its generation.
# * **Generation** - This is where the LLM will generate a response that has been flavoured/augmented with the retrieved resources. In turn, this not only gives us a potentially more correct answer, it also gives us resources to investigate more (since we know which resources went into the prompt).
# 
# The whole idea of RAG is to get an LLM to be more factually correct based on your own input as well as have a reference to where the generated output may have come from.
# 
# This is an incredibly helpful tool.
# 
# Let's say you had 1000s of customer support documents.
# 
# You could use RAG to generate direct answers to questions with links to relevant documentation.
# 
# Or you were an insurance company with large chains of claims emails.
# 
# You could use RAG to answer questions about the emails with sources.
# 
# One helpful analogy is to think of LLMs as calculators for words.
# 
# With good inputs, the LLM can sort them into helpful outputs.
# 
# How?
# 
# It starts with better search.

# %% [markdown]
# ### Similarity search
# 
# Similarity search or semantic search or vector search is the idea of searching on *vibe*.
# 
# If this sounds like woo, woo. It's not.
# 
# Perhaps searching via *meaning* is a better analogy.
# 
# With keyword search, you are trying to match the string "apple" with the string "apple".
# 
# Whereas with similarity/semantic search, you may want to search "macronutrients functions".
# 
# And get back results that don't necessarily contain the words "macronutrients functions" but get back pieces of text that match that meaning.
# 
# > **Example:** Using similarity search on our textbook data with the query "macronutrients function" returns a paragraph that starts with:
# >
# >*There are three classes of macronutrients: carbohydrates, lipids, and proteins. These can be metabolically processed into cellular energy. The energy from macronutrients comes from their chemical bonds. This chemical energy is converted into cellular energy that is then utilized to perform work, allowing our bodies to conduct their basic functions.*
# >
# > as the first result. How cool!
# 
# If you've ever used Google, you know this kind of workflow.
# 
# But now we'd like to perform that across our own data.
# 
# Let's import our embeddings we created earlier (tk -link to embedding file) and prepare them for use by turning them into a tensor.

# %%
import random

import torch
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import texts and embedding df
text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")

# Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# Convert texts and embedding df to list of dicts
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

# Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
embeddings.shape

# %%
text_chunks_and_embedding_df.head()

# %%
embeddings[0]

# %% [markdown]
# Nice!
# 
# Now let's prepare another instance of our embedding model. Not because we have to but because we'd like to make it so you can start the notebook from the cell above.

# %%
from sentence_transformers import util, SentenceTransformer

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device=device) # choose the device to load the model to

# %% [markdown]
# Embedding model ready!
# 
# Time to perform a semantic search.
# 
# Let's say you were studying the macronutrients.
# 
# And wanted to search your textbook for "macronutrients functions".
# 
# Well, we can do so with the following steps:
# 1. Define a query string (e.g. `"macronutrients functions"`) - note: this could be anything, specific or not.
# 2. Turn the query string in an embedding with same model we used to embed our text chunks.
# 3. Perform a [dot product](https://pytorch.org/docs/stable/generated/torch.dot.html) or [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) function between the text embeddings and the query embedding (we'll get to what these are shortly) to get similarity scores.
# 4. Sort the results from step 3 in descending order (a higher score means more similarity in the eyes of the model) and use these values to inspect the texts.
# 
# Easy!
# 

# %%
# 1. Define the query
# Note: This could be anything. But since we're working with a nutrition textbook, we'll stick with nutrition-based queries.
query = "macronutrients functions"
print(f"Query: {query}")

# 2. Embed the query to the same numerical space as the text examples
# Note: It's important to embed your query with the same model you embedded your examples with.
query_embedding = embedding_model.encode(query, convert_to_tensor=True)

# 3. Get similarity scores with the dot product (we'll time this for fun)
from time import perf_counter as timer

start_time = timer()
dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
end_time = timer()

print(f"Time take to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

# 4. Get the top-k results (we'll keep this to 5)
top_results_dot_product = torch.topk(dot_scores, k=1)
top_results_dot_product

# %% [markdown]
# Woah!! Now that was fast!
# 
# ~0.00008 seconds to perform a dot product comparison across 1680 embeddings on my machine (NVIDIA RTX 4090 GPU).
# 
# GPUs are optimized for these kinds of operations.
# 
# So even if you we're to increase our embeddings by 100x (1680 -> 168,000), an exhaustive dot product operation would happen in ~0.008 seconds (assuming linear scaling).
# 
# Heck, let's try it.

# %%
larger_embeddings = torch.randn(100*embeddings.shape[0], 768).to(device)
print(f"Embeddings shape: {larger_embeddings.shape}")

# Perform dot product across 168,000 embeddings
start_time = timer()
dot_scores = util.dot_score(a=query_embedding, b=larger_embeddings)[0]
end_time = timer()

print(f"Time take to get scores on {len(larger_embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

# %% [markdown]
# Wow. That's quick!
# 
# That means we can get pretty far by just storing our embeddings in `torch.tensor` for now.
# 
# However, for *much* larger datasets, we'd likely look at a dedicated vector database/indexing libraries such as [Faiss](https://github.com/facebookresearch/faiss).
# 
# Let's check the results of our original similarity search.
# 
# [`torch.topk`](https://pytorch.org/docs/stable/generated/torch.topk.html) returns a tuple of values (scores) and indicies for those scores.
# 
# The indicies relate to which indicies in the `embeddings` tensor have what scores in relation to the query embedding (higher is better).
# 
# We can use those indicies to map back to our text chunks.
# 
# First, we'll define a small helper function to print out wrapped text (so it doesn't print a whole text chunk as a single line).

# %%
# Define helper function to print wrapped text
import textwrap

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

# %% [markdown]
# Now we can loop through the `top_results_dot_product` tuple and match up the scores and indicies and then use those indicies to index on our `pages_and_chunks` variable to get the relevant text chunk.
# 
# Sounds like a lot but we can do it!

# %%
print(f"Query: '{query}'\n")
print("Results:")
# Loop through zipped together scores and indicies from torch.topk
for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
    print(f"Score: {score:.4f}")
    # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
    print("Text:")
    print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
    # Print the page number too so we can reference the textbook further (and check the results)
    print(f"Page number: {pages_and_chunks[idx]['page_number']}")
    print("\n")

# %% [markdown]
# The first result looks to have nailed it!
# 
# We get a very relevant answer to our query `"macronutrients functions"` even though its quite vague.
# 
# That's the power of semantic search!
# 
# And even better, if we wanted to inspect the result further, we get the page number where the text appears.
# 
# How about we check the page to verify?
# 
# We can do so by loading the page number containing the highest result (page 5 but really page 5 + 41 since our PDF page numbers start on page 41).

# %%
import fitz

# Open PDF and load target page
pdf_path = "human-nutrition-text.pdf" # requires PDF to be downloaded
doc = fitz.open(pdf_path)
page = doc.load_page(5 + 41) # number of page (our doc starts page numbers on page 41)

# Get the image of the page
img = page.get_pixmap(dpi=300)

# Optional: save the image
#img.save("output_filename.png")
doc.close()

# Convert the Pixmap to a numpy array
img_array = np.frombuffer(img.samples_mv,
                          dtype=np.uint8).reshape((img.h, img.w, img.n))

# Display the image using Matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(13, 10))
plt.imshow(img_array)
plt.title(f"Query: '{query}' | Most relevant page:")
plt.axis('off') # Turn off axis
plt.show()

# %% [markdown]
# Nice!
# 
# Now we can do extra research if we'd like.
# 
# We could repeat this workflow for any kind of query we'd like on our textbook.
# 
# And it would also work for other datatypes too.
# 
# We could use semantic search on customer support documents.
# 
# Or email threads.
# 
# Or company plans.
# 
# Or our old journal entries.
# 
# Almost anything!
# 
# The workflow is the same:
# 
# `ingest documents -> split into chunks -> embed chunks -> make a query -> embed the query -> compare query embedding to chunk embeddings`
# 
# And we get relevant resources *along with* the source they came from!
# 
# That's the **retrieval** part of Retrieval Augmented Generation (RAG).
# 
# Before we get to the next two steps, let's take a small aside and discuss similarity measures.

# %% [markdown]
# ### Similarity measures: dot product and cosine similarity
# 
# Let's talk similarity measures between vectors.
# 
# Specifically, embedding vectors which are representations of data with magnitude and direction in high dimensional space (our embedding vectors have 768 dimensions).
# 
# Two of the most common you'll across are the dot product and cosine similarity.
# 
# They are quite similar.
# 
# The main difference is that cosine similarity has a normalization step.
# 
# | Similarity measure | Description | Code |
# | ----- | ----- | ----- |
# | [Dot Product](https://en.wikipedia.org/wiki/Dot_product) | - Measure of magnitude and direction between two vectors<br>- Vectors that are aligned in direction and magnitude have a higher positive value<br>- Vectors that are opposite in direction and magnitude have a higher negative value | [`torch.dot`](https://pytorch.org/docs/stable/generated/torch.dot.html), [`np.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html), [`sentence_transformers.util.dot_score`](https://www.sbert.net/docs/package_reference/util.html#sentence_transformers.util.dot_score) |
# | [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity) | - Vectors get normalized by magnitude/[Euclidean norm](https://en.wikipedia.org/wiki/Norm_(mathematics))/L2 norm so they have unit length and are compared more so on direction<br>- Vectors that are aligned in direction have a value close to 1<br>- Vectors that are opposite in direction have a value close to -1 | [`torch.nn.functional.cosine_similarity`](https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html), [`1 - scipy.spatial.distance.cosine`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html) (subtract the distance from 1 for similarity measure), [`sentence_transformers.util.cos_sim`](https://www.sbert.net/docs/package_reference/util.html#sentence_transformers.util.cos_sim) |
# 
# For text similarity, you generally want to use cosine similarity as you are after the semantic measurements (direction) rather than magnitude.
# 
# In our case, our embedding model `all-mpnet-base-v2` outputs normalized outputs (see the [Hugging Face model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2#usage-huggingface-transformers) for more on this) so dot product and cosine similarity return the same results. However, dot product is faster due to not need to perform a normalize step.
# 
# To make things bit more concrete, let's make simple dot product and cosine similarity functions and view their results on different vectors.
# 
# > **Note:** Similarity measures between vectors and embeddings can be used on any kind of embeddings, not just text embeddings. For example, you could measure image embedding similarity or audio embedding similarity. Or with text and image models like [CLIP](https://github.com/mlfoundations/open_clip), you can measure the similarity between text and image embeddings.

# %%
import torch

def dot_product(vector1, vector2):
    return torch.dot(vector1, vector2)

def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)

    # Get Euclidean/L2 norm of each vector (removes the magnitude, keeps direction)
    norm_vector1 = torch.sqrt(torch.sum(vector1**2))
    norm_vector2 = torch.sqrt(torch.sum(vector2**2))

    return dot_product / (norm_vector1 * norm_vector2)

# Example tensors
vector1 = torch.tensor([1, 2, 3], dtype=torch.float32)
vector2 = torch.tensor([1, 2, 3], dtype=torch.float32)
vector3 = torch.tensor([4, 5, 6], dtype=torch.float32)
vector4 = torch.tensor([-1, -2, -3], dtype=torch.float32)

# Calculate dot product
print("Dot product between vector1 and vector2:", dot_product(vector1, vector2))
print("Dot product between vector1 and vector3:", dot_product(vector1, vector3))
print("Dot product between vector1 and vector4:", dot_product(vector1, vector4))

# Calculate cosine similarity
print("Cosine similarity between vector1 and vector2:", cosine_similarity(vector1, vector2))
print("Cosine similarity between vector1 and vector3:", cosine_similarity(vector1, vector3))
print("Cosine similarity between vector1 and vector4:", cosine_similarity(vector1, vector4))

# %% [markdown]
# Notice for both dot product and cosine similarity the comparisons of `vector1` and `vector2` are the opposite of `vector1` and `vector4`.
# 
# Comparing `vector1` and `vector2` both equations return positive values (14 for dot product and 1.0 for cosine similarity).
# 
# But comparing `vector1` and `vector4` the result is in the negative direction.
# 
# This makes sense because `vector4` is the negative version of `vector1`.
# 
# Whereas comparing `vector1` and `vector3` shows a different outcome.
# 
# For the dot product, the value is positive and larger then the comparison of two exactly the same vectors (32 vs 14).
# 
# However, for the cosine similarity, thanks to the normalization step, comparing `vector1` and `vector3` results in a postive value close to 1 but not exactly 1.
# 
# It is because of this that when comparing text embeddings, cosine similarity is generally favoured as it measures the difference in direction of a pair of vectors rather than difference in magnitude.
# 
# And it is this difference in direction that is more generally considered to capture the semantic meaning/vibe of the text.
# 
# The good news is that as mentioned before, the outputs of our embedding model `all-mpnet-base-v2` are already normalized.
# 
# So we can continue using the dot product (cosine similarity is dot product + normalization).
# 
# With similarity measures explained, let's functionize our semantic search steps from above so we can repeat them.

# %% [markdown]
# ### Functionizing our semantic search pipeline
# 
# Let's put all of the steps from above for semantic search into a function or two so we can repeat the workflow.

# %%
def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5,
                                print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query,
                                   convert_to_tensor=True)

    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    scores, indices = torch.topk(input=dot_scores,
                                 k=n_resources_to_return)

    return scores, indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict]=pages_and_chunks,
                                 n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """

    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)

    print(f"Query: {query}\n")
    print("Results:")
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print_wrapped(pages_and_chunks[index]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further and check the results
        print(f"Page number: {pages_and_chunks[index]['page_number']}")
        print("\n")

# %% [markdown]
# Excellent! Now let's test our functions out.

# %%
query = "symptoms of pellagra"

# Get just the scores and indices of top related results
scores, indices = retrieve_relevant_resources(query=query,
                                              embeddings=embeddings)
scores, indices

# %%
# Print out the texts of the top scores
print_top_results_and_scores(query=query,
                             embeddings=embeddings)

# %% [markdown]
# ### Semantic search/vector search extensions
# 
# We've covered an exmaple of using embedding vector search to find relevant results based on a query.
# 
# However, you could also add to this pipeline with traditional keyword search.
# 
# Many modern search systems use keyword and vector search in tandem.
# 
# Our dataset is small and allows for an exhaustive search (comparing the query to *every* possible result) but if you start to work with large scale datasets with hundred of thousands, millions or even billions of vectors, you'll want to implement an index.
# 
# You can think of an index as sorting your embeddings before you search through them.
# 
# So it narrows down the search space.
# 
# For example, it would be inefficient to search every word in the dictionary to find the word "duck", instead you'd go straight to the letter D, perhaps even straight to the back half of the letter D, find words close to "duck" before finding it.
# 
# That's how an index can help search through many examples without comprimising too much on speed or quality (for more on this, check out [nearest neighbour search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)).
# 
# One of the most popular indexing libraries is [Faiss](https://github.com/facebookresearch/faiss).
# 
# Faiss is open-source and was originally created by Facebook to deal with internet-scale vectors and implements many algorithms such as [HNSW](https://arxiv.org/abs/1603.09320) (Hierarchical Naviganle Small Worlds).

# %% [markdown]
# ### Getting an LLM for local generation
# 
# We're got our retrieval pipeline ready, let's now get the generation side of things happening.
# 
# To perform generation, we're going to use a Large Language Model (LLM).
# 
# LLMs are designed to generate an output given an input.
# 
# In our case, we want our LLM to generate and output of text given a input of text.
# 
# And more specifically, we want the output of text to be generated based on the context of relevant information to the query.
# 
# The input to an LLM is often referred to as a prompt.
# 
# We'll augment our prompt with a query as well as context from our textbook related to that query.
# 
# > **Which LLM should I use?**
# 
# There are many LLMs available.
# 
# Two of the main questions to ask from this is:
# 1. Do I want it to run locally?
# 2. If yes, how much compute power can I dedicate?
# 
# If you're after the absolute best performance, you'll likely want to use an API (not running locally) such as GPT-4 or Claude 3. However, this comes with the tradeoff of sending your data away and then awaiting a response.
# 
# For our case, since we want to set up a local pipeline and run it on our own GPU, we'd answer "yes" to the first question and then the second question will depend on what hardware we have available.
# 
# To find open-source LLMs, one great resource is the [Hugging Face open LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).
# 
# The leaderboard compares many of the latest and greatest LLMs on various benchmarks.
# 
# Another great resource is [TheBloke on Hugging Face](https://huggingface.co/TheBloke), an account which provides an extensive range of quantized (models that have been made smaller) LLMs.
# 
# A rule of thumb for LLMs (and deep learning models in general) is that the higher the number of parameters, the better the model performs.
# 
# It may be tempting to go for the largest size model (e.g. a 70B parameter model rather than a 7B parameter model) but a larger size model may not be able to run on your available hardware.
# 
# The following table gives an insight into how much GPU memory you'll need to load an LLM with different sizes and different levels of [numerical precision](https://en.wikipedia.org/wiki/Precision_(computer_science)).
# 
# They are based on the fact that 1 float32 value (e.g. `0.69420`) requires 4 bytes of memory and 1GB is approximately 1,000,000,000 (one billion) bytes.
# 
# | Model Size (Billion Parameters) | Float32 VRAM (GB) | Float16 VRAM (GB) | 8-bit VRAM (GB) | 4-bit VRAM (GB) |
# |-----|-----|-----|-----|-----|
# | 1B                              | ~4                | ~2                | ~1              | ~0.5            |
# | 7B (e.g., [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b), [Gemma 7B](https://huggingface.co/google/gemma-7b-it), [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1))             | ~28               | ~14               | ~7              | ~3.5            |
# | 10B                             | ~40               | ~20               | ~10             | ~5              |
# | 70B (e.g, Llama 2 70B)          | ~280              | ~140              | ~70             | ~35             |
# | 100B                            | ~400              | ~200              | ~100            | ~50             |
# | 175B                            | ~700              | ~350              | ~175            | ~87.5           |
# 
# <br>
# 
# > **Note:** Loading a model in a lower precision (e.g. 8-bit instead of float16) generally lowers performance. Lower precision can help to reduce computing requirements, however sometimes the performance degradation in terms of model output can be substantial. Finding the right speed/performance tradeoff will often require many experiments.

# %% [markdown]
# ### Checking local GPU memory availability
# 
# Let's find out what hardware we've got available and see what kind of model(s) we'll be able to load.
# 
# > **Note:** You can also check this with the `!nvidia-smi` command.

# %%
# Get GPU available memory
import torch
gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
gpu_memory_gb = round(gpu_memory_bytes / (2**30))
print(f"Available GPU memory: {gpu_memory_gb} GB")

# %% [markdown]
# Ok wonderful!
# 
# I'm running this notebook with a NVIDIA RTX 4090, so I've got 24GB of VRAM available.
# 
# However, this may be different on your end.
# 
# Looking at the table above, it seems we can run a ~7-10B parameter model in float16 precision pretty comfortably.
# 
# But we could also run a smaller one if we'd like.
# 
# Let's try out the recently released (at the time of writing, March 2024) LLM from Google, [Gemma](https://huggingface.co/blog/gemma).
# 
# Specifically, we'll use the `gemma-7b-it` version which stands for Gemma 7B Instruction-Tuned.
# 
# Instruction tuning is the process of tuning a raw language model to follow instructions.
# 
# These are the kind of models you'll find in most chat-based assistants such as ChatGPT, Gemini or Claude.
# 
# The following table shows different amounts of GPU memory requirements for different verions of the Gemma LLMs with varying levels of precision.
# 
# | Model             | Precision | Min-Memory (Bytes) | Min-Memory (MB) | Min-Memory (GB) | Recommended Memory (GB) | Hugging Face ID |
# |-------------------|-----------|----------------|-------------|-------------| ----- | ----- |
# | [Gemma 2B](https://huggingface.co/google/gemma-2b-it)          | 4-bit     | 2,106,749,952  | 2009.15     | 1.96        | ~5.0 | [`gemma-2b`](https://huggingface.co/google/gemma-2b) or [`gemma-2b-it`](https://huggingface.co/google/gemma-2b-it) for instruction tuned version |
# | Gemma 2B          | Float16   | 5,079,453,696  | 4844.14     | 4.73        | ~8.0 | Same as above |
# | [Gemma 7B](https://huggingface.co/google/gemma-7b-it)          | 4-bit     | 5,515,859,968  | 5260.33     | 5.14        | ~8.0 | [`gemma-7b`](https://huggingface.co/google/gemma-7b) or [`gemma-7b-it`](https://huggingface.co/google/gemma-7b-it) for instruction tuned version |
# | Gemma 7B          | Float16   | 17,142,470,656 | 16348.33    | 15.97       | ~19 | Same as above |
# 
# > **Note:** `gemma-7b-it` means "instruction tuned", as in, a base LLM (`gemma-7b`) has been fine-tuned to follow instructions, similar to [`Mistral-7B-v0.1`](https://huggingface.co/mistralai/Mistral-7B-v0.1) and [`Mistral-7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).
# >
# > There are also further quantized and smaller variants of Gemma (and other LLMs) available in various formats such as GGUF. You can see many of these on [TheBloke account on Hugging Face](https://huggingface.co/TheBloke).
# >
# > The version of LLM you choose to use will be largely based on project requirements and experimentation.
# 
# Based on the table above, let's write a simple if/else statement which recommends which Gemma variant we should look into using.

# %%
# Note: the following is Gemma focused, however, there are more and more LLMs of the 2B and 7B size appearing for local use.
if gpu_memory_gb < 5.1:
    print(f"Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
elif gpu_memory_gb < 8.1:
    print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
    use_quantization_config = True
    model_id = "google/gemma-2b-it"
elif gpu_memory_gb < 19.0:
    print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
    use_quantization_config = False
    model_id = "google/gemma-2b-it"
elif gpu_memory_gb > 19.0:
    print(f"GPU memory: {gpu_memory_gb} | Recommend model: Gemma 7B in 4-bit or float16 precision.")
    use_quantization_config = False
    model_id = "google/gemma-7b-it"

print(f"use_quantization_config set to: {use_quantization_config}")
print(f"model_id set to: {model_id}")

# %% [markdown]
# ### Loading an LLM locally
# 
# Alright! Looks like `gemma-7b-it` it is (for my local machine with an RTX 4090, change the `model_id` and `use_quantization_config` values to suit your needs)!
# 
# There are plenty of examples of how to load the model on the `gemma-7b-it` [Hugging Face model card](https://huggingface.co/google/gemma-7b-it).
# 
# Good news is, the Hugging Face [`transformers`](https://huggingface.co/docs/transformers/) library has all the tools we need.
# 
# To load our LLM, we're going to need a few things:
# 1. A quantization config (optional) - This will determine whether or not we load the model in 4bit precision for lower memory usage. The we can create this with the [`transformers.BitsAndBytesConfig`](https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/quantization#transformers.BitsAndBytesConfig) class (requires installing the [`bitsandbytes` library](https://github.com/TimDettmers/bitsandbytes)).
# 2. A model ID - This is the reference Hugging Face model ID which will determine which tokenizer and model gets used. For example `gemma-7b-it`.
# 3. A tokenzier - This is what will turn our raw text into tokens ready for the model. We can create it using the [`transformers.AutoTokenzier.from_pretrained`](https://huggingface.co/docs/transformers/v4.38.2/en/model_doc/auto#transformers.AutoTokenizer) method and passing it our model ID.
# 4. An LLM model - Again, using our model ID we can load a specific LLM model. To do so we can use the [`transformers.AutoModelForCausalLM.from_pretrained`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) method and passing it our model ID as well as other various parameters.
# 
# As a bonus, we'll check if [Flash Attention 2](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2) is available using `transformers.utils.is_flash_attn_2_available()`. Flash Attention 2 speeds up the attention mechanism in Transformer architecture models (which is what many modern LLMs are based on, including Gemma). So if it's available and the model is supported (not all models support Flash Attention 2), we'll use it. If it's not available, you can install it by following the instructions on the [GitHub repo](https://github.com/Dao-AILab/flash-attention).
# 
# > **Note:** Flash Attention 2 currently works on NVIDIA GPUs with a compute capability score of 8.0+ (Ampere, Ada Lovelace, Hopper architectures). We can check our GPU compute capability score with [`torch.cuda.get_device_capability(0)`](https://pytorch.org/docs/stable/generated/torch.cuda.get_device_capability.html).
# 
# > **Note:** To get access to the Gemma models, you will have to [agree to the terms & conditions](https://huggingface.co/google/gemma-7b-it) on the Gemma model page on Hugging Face. You will then have to authorize your local machine via the [Hugging Face CLI/Hugging Face Hub `login()` function](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication). Once you've done this, you'll be able to download the models. If you're using Google Colab, you can add a [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) to the "Secrets" tab.
# >
# > Downloading an LLM locally can take a fair bit of time depending on your internet connection. Gemma 7B is about a 16GB download and Gemma 2B is about a 6GB download.
# 
# Let's do it!

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available


# Bonus: Setup Flash Attention 2 for faster inference, default to "sdpa" or "scaled dot product attention" if it's not available
# Flash Attention 2 requires NVIDIA GPU compute capability of 8.0 or above, see: https://developer.nvidia.com/cuda-gpus
# Requires !pip install flash-attn, see: https://github.com/Dao-AILab/flash-attention
if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
  attn_implementation = "flash_attention_2"
else:
  attn_implementation = "sdpa"
print(f"[INFO] Using attention implementation: {attn_implementation}")

# 2. Pick a model we'd like to use (this will depend on how much GPU memory you have available)
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_id = model_id # (we already set this above)
print(f"[INFO] Using model_id: {model_id}")


# 3. Instantiate tokenizer (tokenizer turns text into numbers ready for the model)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id, cache_dir="D:\HuggingFace")

# 4. Instantiate the model
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                 cache_dir="D:\HuggingFace",
                                                 torch_dtype=torch.float16, # datatype to use, we want float16
                                                 low_cpu_mem_usage=False, # use full memory
                                                 attn_implementation=attn_implementation) # which attention version to use

if not use_quantization_config: # quantization takes care of device setting automatically, so if it's not used, send model to GPU
    llm_model.to("cuda")

# %% [markdown]
# We've got an LLM!
# 
# Let's check it out.

# %%
llm_model

# %% [markdown]
# Ok, ok a bunch of layers ranging from embedding layers to attention layers (see the `GemmaFlashAttention2` layers!) to MLP and normalization layers.
# 
# The good news is that we don't have to know too much about these to use the model.
# 
# How about we get the number of parameters in our model?

# %%
def get_model_num_params(model: torch.nn.Module):
    return sum([param.numel() for param in model.parameters()])

get_model_num_params(llm_model)

# %% [markdown]
# Hmm, turns out that Gemma 7B is really Gemma 8.5B.
# 
# It pays to do your own investigations!
# 
# How about we get the models memory requirements?

# %%
def get_model_mem_size(model: torch.nn.Module):
    """
    Get how much memory a PyTorch model takes up.

    See: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
    """
    # Get model parameters and buffer sizes
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate various model sizes
    model_mem_bytes = mem_params + mem_buffers # in bytes
    model_mem_mb = model_mem_bytes / (1024**2) # in megabytes
    model_mem_gb = model_mem_bytes / (1024**3) # in gigabytes

    return {"model_mem_bytes": model_mem_bytes,
            "model_mem_mb": round(model_mem_mb, 2),
            "model_mem_gb": round(model_mem_gb, 2)}

get_model_mem_size(llm_model)

# %% [markdown]
# Nice, looks like this model takes up 15.97GB of space on the GPU.
# 
# Plus a little more for the forward pass (due to all the calculations happening between the layers).
# 
# Hence why I rounded it up to be ~19GB in the table above.
# 
# Now let's get to the fun part, generating some text!

# %% [markdown]
# ### Generating text with our LLM
# 
# We can generate text with our LLM `model` instance by calling the [`generate()` method](https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/text_generation#transformers.GenerationConfig) (this method has plenty of options to pass into it alongside the text) on it and passing it a tokenized input.
# 
# The tokenized input comes from passing a string of text to our `tokenizer`.
# 
# It's important to note that you should use a tokenizer that has been paired with a model.
# 
# Otherwise if you try to use a different tokenizer and then pass those inputs to a model, you will likely get errors/strange results.
# 
# For some LLMs, there's a specific template you should pass to them for ideal outputs.
# 
# For example, the `gemma-7b-it` model has been trained in a dialogue fashion (instruction tuning).
# 
# In this case, our `tokenizer` has a [`apply_chat_template()` method](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template) which can prepare our input text in the right format for the model.
# 
# Let's try it out.
# 
# > **Note:** The following demo has been modified from the Hugging Face model card for [Gemma 7B](https://huggingface.co/google/gemma-7b-it). Many similar demos of usage are available on the model cards of similar models.

# %%
input_text = "What are the macronutrients, and what roles do they play in the human body?"
print(f"Input text:\n{input_text}")

# Create prompt template for instruction-tuned model
dialogue_template = [
    {"role": "user",
     "content": input_text}
]

# Apply the chat template
prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                       tokenize=False, # keep as raw text (not tokenized)
                                       add_generation_prompt=True)
print(f"\nPrompt (formatted):\n{prompt}")

# %% [markdown]
# Notice the scaffolding around our input text, this is the kind of turn-by-turn instruction tuning our model has gone through.
# 
# Our next step is to tokenize this formatted text and pass it to our model's `generate()` method.
# 
# We'll make sure our tokenized text is on the same device as our model (GPU) using `to("cuda")`.
# 
# Let's generate some text!
# 
# We'll time it for fun with the `%%time` magic.

# %%
%%time

# Tokenize the input text (turn it into numbers) and send it to GPU
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
print(f"Model input (tokenized):\n{input_ids}\n")

# Generate outputs passed on the tokenized input
# See generate docs: https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/text_generation#transformers.GenerationConfig
outputs = llm_model.generate(**input_ids,
                             max_new_tokens=256) # define the maximum number of new tokens to create
print(f"Model output (tokens):\n{outputs[0]}\n")

# %% [markdown]
# Woohoo! We just generated some text on our local GPU!
# 
# Well not just yet...
# 
# Our LLM accepts tokens in and sends tokens back out.
# 
# We can conver the output tokens to text using [`tokenizer.decode()`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.decode).

# %%
# Decode the output tokens to text
outputs_decoded = tokenizer.decode(outputs[0])
print(f"Model output (decoded):\n{outputs_decoded}\n")

# %% [markdown]
# Woah! That looks like a pretty good answer.
# 
# But notice how the output contains the prompt text as well?
# 
# How about we do a little formatting to replace the prompt in the output text?
# 
# > **Note:** `"<bos>"` and `"<eos>"` are special tokens to denote "beginning of sentence" and "end of sentence" respectively.

# %%
print(f"Input text: {input_text}\n")
print(f"Output text:\n{outputs_decoded.replace(prompt, '').replace('<bos>', '').replace('<eos>', '')}")

# %% [markdown]
# How cool is that!
# 
# We just officially generated text from an LLM running locally.
# 
# So we've covered the R (retrieval) and G (generation) of RAG.
# 
# How about we check out the last step?
# 
# Augmentation.
# 
# First, let's put together a list of queries we can try out with our pipeline.

# %%
# Nutrition-style questions generated with GPT4
gpt4_questions = [
    "What are the macronutrients, and what roles do they play in the human body?",
    "How do vitamins and minerals differ in their roles and importance for health?",
    "Describe the process of digestion and absorption of nutrients in the human body.",
    "What role does fibre play in digestion? Name five fibre containing foods.",
    "Explain the concept of energy balance and its importance in weight management."
]

# Manually created question list
manual_questions = [
    "How often should infants be breastfed?",
    "What are symptoms of pellagra?",
    "How does saliva help with digestion?",
    "What is the RDI for protein per day?",
    "water soluble vitamins"
]

query_list = gpt4_questions + manual_questions

# %% [markdown]
# And now let's check if our `retrieve_relevant_resources()` function works with our list of queries.

# %%
import random
query = random.choice(query_list)

print(f"Query: {query}")

# Get just the scores and indices of top related results
scores, indices = retrieve_relevant_resources(query=query,
                                              embeddings=embeddings)
scores, indices

# %% [markdown]
# Beautiful!
# 
# Let's augment!

# %% [markdown]
# ### Augmenting our prompt with context items
# 
# What we'd like to do with augmentation is take the results from our search for relevant resources and put them into the prompt that we pass to our LLM.
# 
# In essence, we start with a base prompt and update it with context text.
# 
# Let's write a function called `prompt_formatter` that takes in a query and our list of context items (in our case it'll be select indices from our list of dictionaries inside `pages_and_chunks`) and then formats the query with text from the context items.
# 
# We'll apply the dialogue and chat template to our prompt before returning it as well.
# 
# > **Note:** The process of augmenting or changing a prompt to an LLM is known as prompt engineering. And the best way to do it is an active area of research. For a comprehensive guide on different prompt engineering techniques, I'd recommend the Prompt Engineering Guide ([promptingguide.ai](https://www.promptingguide.ai/)), [Brex's Prompt Engineering Guide](https://github.com/brexhq/prompt-engineering) and the paper [Prompt Design and Engineering: Introduction and Advanced Models](https://arxiv.org/abs/2401.14423).

# %%
def prompt_formatter(query: str,
                     context_items: list[dict]) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt

# %% [markdown]
# Looking good! Let's try our function out.

# %%
query = random.choice(query_list)
print(f"Query: {query}")

# Get relevant resources
scores, indices = retrieve_relevant_resources(query=query,
                                              embeddings=embeddings)

# Create a list of context items
context_items = [pages_and_chunks[i] for i in indices]

# Format prompt with context items
prompt = prompt_formatter(query=query,
                          context_items=context_items)
print(prompt)

# %% [markdown]
# What a good looking prompt!
# 
# We can tokenize this and pass it straight to our LLM.

# %%
%%time

input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate an output of tokens
outputs = llm_model.generate(**input_ids,
                             temperature=0.7, # lower temperature = more deterministic outputs, higher temperature = more creative outputs
                             do_sample=True, # whether or not to use sampling, see https://huyenchip.com/2024/01/16/sampling.html for more
                             max_new_tokens=256) # how many new tokens to generate from prompt

# Turn the output tokens into text
output_text = tokenizer.decode(outputs[0])

print(f"Query: {query}")
print(f"RAG answer:\n{output_text.replace(prompt, '')}")

# %% [markdown]
# Yesssssss!!!
# 
# Our RAG pipeline is complete!
# 
# We just Retrieved, Augmented and Generated!
# 
# And all on our own local GPU!
# 
# How about we functionize the generation step to make it easier to use?
# 
# We can put a little formatting on the text being returned to make it look nice too.
# 
# And we'll make an option to return the context items if needed as well.

# %%
def ask(query,
        temperature=0.7,
        max_new_tokens=512,
        format_answer_text=True,
        return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """

    # Get just the scores and indices of top related results
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings)

    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context item
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu() # return score back to CPU

    # Format the prompt with context items
    prompt = prompt_formatter(query=query,
                              context_items=context_items)

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)

    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

    # Only return the answer without the context items
    if return_answer_only:
        return output_text

    return output_text, context_items

# %% [markdown]
# What a good looking function!
# 
# The workflow could probably be a little refined but this should work!
# 
# Let's try it out.

# %%
query = random.choice(query_list)
print(f"Query: {query}")

# Answer query with context and return context
answer, context_items = ask(query=query,
                            temperature=0.7,
                            max_new_tokens=512,
                            return_answer_only=False)

print(f"Answer:\n")
print_wrapped(answer)
print(f"Context items:")
context_items

# %% [markdown]
# Local RAG workflow complete!
# 
# We've now officially got a way to Retrieve, Augment and Generate answers based on a source.
# 
# For now we can verify our answers manually by reading them and reading through the textbook.
# 
# But if you want to put this into a production system, it'd be a good idea to have some kind of evaluation on how well our pipeline works.
# 
# For example, you could use another LLM to rate the answers returned by our LLM and then use those ratings as a proxy evaluation.
# 
# However, I'll leave this and a few more interesting ideas as extensions.

# %% [markdown]
# ## Extensions
# 
# * May want to improve text extraction with something like Marker - https://github.com/VikParuchuri/marker
# * Guide to more advanced PDF extraction - https://towardsdatascience.com/extracting-text-from-pdf-files-with-python-a-comprehensive-guide-9fc4003d517
# * See the following prompt engineering resources for more prompting techniques - promptinguide.ai, Brex's Prompt Engineering Guide
# * What happens when a query comes through that there isn't any context in the textbook on?
# * Try another embedding model (e.g. Mixed Bread AI large, `mixedbread-ai/mxbai-embed-large-v1`, see: https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
# * Try another LLM... (e.g. Mistral-Instruct)
# * Try different prompts (e.g. see prompting techniques online)
# * Our example only focuses on text from a PDF, however, we could extend it to include figures and images
# * Evaluate the answers -> could use another LLM to rate our answers (e.g. use GPT-4 to make)
# * Vector database/index for larger setup (e.g. 100,000+ chunks)
# * Libraries/frameworks such as LangChain / LlamaIndex can help do many of the steps for you - so it's worth looking into those next, wanted to recreate a workflow with lower-level tools to show the principles
# * Optimizations for speed
#     * See Hugging Face docs for recommended speed ups on GPU - https://huggingface.co/docs/transformers/perf_infer_gpu_one
#     * Optimum NVIDIA - https://huggingface.co/blog/optimum-nvidia, GitHub: https://github.com/huggingface/optimum-nvidia
#     * See NVIDIA TensorRT-LLM - https://github.com/NVIDIA/TensorRT-LLM
#     * See GPT-Fast for PyTorch-based optimizations - https://github.com/pytorch-labs/gpt-fast
#     * Flash attention 2 (requires Ampere GPUs or newer) - https://github.com/Dao-AILab/flash-attention
# * Stream text output so it looks prettier (e.g. each token appears as it gets output from the model)
# * Turn the workflow into an app, see Gradio type chatbots for this - https://www.gradio.app/guides/creating-a-chatbot-fast, see local example: https://www.gradio.app/guides/creating-a-chatbot-fast#example-using-a-local-open-source-llm-with-hugging-face


