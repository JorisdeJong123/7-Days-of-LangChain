"""
This script shows how to create a mindmap based on your study material.
We're using an easy LangChain implementation to show how to use the different components of LangChain.

Once you have your markdown mindmap, import it to Xmind to create a mindmap.
This is part of my '7 Days of LangChain' series. 

Check out the explanation about the code on my Twitter (@JorisTechTalk)

"""

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback

# Set your OpenAI API Key.
openai_api_key = 'YOUR_API_KEY_HERE'

# Set file path
file_path = 'eight.pdf'

# Load Data from PDF for Question Generation
loader_mindmap = PdfReader(file_path)

# Store all the text in a variable
text = ""
for page in loader_mindmap.pages:
    text += page.extract_text()

# Split Data For Mindmap Generation
text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=10000, chunk_overlap=1000)
texts_for_mindmap = text_splitter.split_text(text)
docs_for_mindmap = [Document(page_content=t) for t in texts_for_mindmap]

# Template for the question generation for every document

prompt_template_mindmap = """

You are an experienced assistant in helping people understand topics through the help of mind maps.

You are an expert in the field of the requested topic.

Make a mindmap based on the context below. Try to make connections between the different topics and be concise.:

------------
{text}
------------

Think step by step.

Always answer in markdown text. Adhere to the following structure:

## Main Topic 1

### Subtopic 1
- Subtopic 1
    -Subtopic 1
    -Subtopic 2
    -Subtopic 3

### Subtopic 2
- Subtopic 1
    -Subtopic 1
    -Subtopic 2
    -Subtopic 3

## Main Topic 2

### Subtopic 1
- Subtopic 1
    -Subtopic 1
    -Subtopic 2
    -Subtopic 3

Make sure you only put out the Markdown text, do not put out anything else. Also make sure you have the correct indentation.


MINDMAP IN MARKDOWN:

"""

PROMPT_MINDMAP = PromptTemplate(template=prompt_template_mindmap, input_variables=["text"])

# Template for refining the mindmap

refine_template_mindmap = ("""

You are an experienced assistant in helping people understand topics through the help of mind maps.

You are an expert in the field of the requested topic.

We have received some mindmap in markdown to a certain extent: {existing_answer}.
We have the option to refine the existing mindmap or add new parts. Try to make connections between the different topics and be concise.
(only if necessary) with some more context below
"------------\n"
"{text}\n"
"------------\n"


Always answer in markdown text. Try to make connections between the different topics and be concise. Adhere to the following structure:

## Main Topic 1

### Subtopic 1
- Subtopic 1
    -Subtopic 1
    -Subtopic 2
    -Subtopic 3

### Subtopic 2
- Subtopic 1
    -Subtopic 1
    -Subtopic 2
    -Subtopic 3

## Main Topic 2

### Subtopic 1
- Subtopic 1
    -Subtopic 1
    -Subtopic 2
    -Subtopic 3

Make sure you only put out the Markdown text, do not put out anything else. Also make sure you have the correct indentation.

MINDMAP IN MARKDOWN:
"""
)
                             
REFINE_PROMPT_MINDMAP = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_mindmap,
)

# Tracking cost
with get_openai_callback() as cb:

    # Initialize the LLM
    llm_markdown = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.3, model="gpt-3.5-turbo-16k")

    # Initialize the summarization chain
    summarize_chain = load_summarize_chain(llm=llm_markdown, chain_type="refine", verbose=True, question_prompt=PROMPT_MINDMAP, refine_prompt=REFINE_PROMPT_MINDMAP)

    # Generate mindmap
    mindmap = summarize_chain(docs_for_mindmap)

    # Save mindmap to .md file
    with open("mindmap.md", "w") as f:
        f.write(mindmap['output_text'])

# Print cost
print(cb)