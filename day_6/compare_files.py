from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import os

# Create a custom input schema
class DocumentInput(BaseModel):
    question: str = Field()

# Set OpenAI API key as environment variable
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# List of files you want to compare
files = [
    {
        "name": "Volkswagen-earnings-Q1-2023",
        "path": "files/Volkswagen-Q1_2023.pdf"
    },
    {
        "name": "tesla-earning-Q1-2023",
        "path": "files/TSLA-Q1-2023-Update.pdf"
    },
]

# Initialize a list of tools
tools = []

# Initialize the LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# Loop over the files
for file in files:
    # Load the documents
    loader = PyPDFLoader(file["path"])
    pages = loader.load_and_split()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    print(f"Loaded {len(docs)} documents from {file['name']}")

    # Vectorize the documents and create a retriever
    embeddings = OpenAIEmbeddings()
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()
    
    # Wrap retrievers in a Tool
    tools.append(
        Tool(
            args_schema=DocumentInput,
            name=file["name"], 
            description=f"useful when you want to answer questions about {file['name']}",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        )
    )

# Initialize LLM for the agent
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-0613", 
)

# Initialize the agent
agent = initialize_agent(
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=tools,
    llm=llm,
    verbose=True,
)

# Initialize the question variable
question = ""

# Run a loop to ask questions
while True and question != "exit":
    question = input("Ask a question or write exit to quit: ")
    if question == "exit":
        break
    answer = agent({"input": question})
    print(answer["output"])
    print("------")