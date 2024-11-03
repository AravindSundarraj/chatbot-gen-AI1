#from langchain.document_loaders import TextLoader
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# TextLoader is class
loader = TextLoader("document.txt")
# data is list of Document and Document is a class with two variables(page_content,metadata)
data = loader.load()
#print(type(data))
#print(data[0].page_content)
csvLoader = CSVLoader("movies.csv")
csvData = csvLoader.load()
#print("Csv-data {} length {} ", csvData, len(csvData))

# Text splitter , we always need text splitter as any LLM will have token limit.

splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

chunks = splitter.split_text(data[0].page_content)
print("the chunk length ", len(chunks), chunks)

for chunk in chunks:
    print(len(chunk))

#####################################################
splitters = RecursiveCharacterTextSplitter(separators=["\n \n", "\n", ".", " "], chunk_size=200, chunk_overlap=0)

chunkss = splitters.split_text(data[0].page_content)
print("the chunkss length ", len(chunkss), chunkss)

for chunki in chunkss:
    print(len(chunki))

# FAISS - a lightweight in-memory vector database type of thing
# FAISS - stands for Facebook similarity search
# https://faiss.ai/

pd.set_option('display.max_colwidth',100)
df = pd.read_csv("sample_text.csv")
print("the shape print   " ,df.shape)
encoder = SentenceTransformer("all-mpnet-base-v2")
vectors = encoder.encode(df.text)
print(vectors.shape)
dim = vectors.shape[1]
print("dim======>>>>>",dim)
#euclidean distance
index = faiss.IndexFlatL2(dim)
index.add(vectors)
prompt = "I want to buy a t-shirt"
veci = encoder.encode(prompt)
# this search expects two dimensional array so convert this vector in to two dimensional array
svec = np.array(veci).reshape(1,-1)
print(svec.shape)
distance,i = index.search(svec,k=2) # k is returning how many similar vectors
#print("distance ===>>> ",distances)
print("i ===>>> ",i)
print("lggg =>> ",df.loc[i[0]])

# convert this single dimension array in two dimensional array







