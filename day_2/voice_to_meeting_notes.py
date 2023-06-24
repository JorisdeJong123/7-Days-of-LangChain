"""
This script shows how to create a meeting notes based on your recordings.
We're using an easy LangChain implementation to show how to use the different components of LangChain.
Also includes an integration with OpenAI Whisper.

This is part of my '7 Days of LangChain' series. 
Check out the explanation about the code on my Twitter (@JorisTechTalk)

"""

import openai
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

# Set your API key
openai_api_key = 'YOUR_API_KEY_HERE'
# os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY_HERE'

# Set your media file path
media_file_path = "meeting_chunk0.mp3"

# Open the media file
media_file = open(media_file_path, "rb")

# Set your model ID
model_id = "whisper-1"

# Call the API
response = openai.Audio.transcribe(
    api_key=openai_api_key,
    model=model_id,
    file=media_file
)

# Assign the transcript to a variable
transcript = response["text"]

# Split the text
text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=10000, chunk_overlap=300)
texts = text_splitter.split_text(transcript)

# Create documents for further processing
docs = [Document(page_content=t) for t in texts]

# Create the prompts

prompt_template_summary = """
You are a management assistant with a specialization in note taking. You are taking notes for a meeting.

Write a detailed summary of the following transcript of a meeting:


{text}

Make sure you don't lose any important information. Be as detailed as possible in your summary. 

Also end with a list of:

- Main takeaways
- Action items
- Decisions
- Open questions
- Next steps

If there are any follow-up meetings, make sure to include them in the summary and mentioned it specifically.


DETAILED SUMMARY IN ENGLISH:"""
PROMPT_SUMMARY = PromptTemplate(template=prompt_template_summary, input_variables=["text"])
refine_template_summary = (
'''
You are a management assistant with a specialization in note taking. You are taking notes for a meeting.
Your job is to provide detailed summary of the following transcript of a meeting:
We have provided an existing summary up to a certain point: {existing_answer}.
We have the opportunity to refine the existing summary (only if needed) with some more context below.
----------------
{text}
----------------
Given the new context, refine the original summary in English.
If the context isn't useful, return the original summary. Make sure you are detailed in your summary.
Make sure you don't lose any important information. Be as detailed as possible. 

Also end with a list of:

- Main takeaways
- Action items
- Decisions
- Open questions
- Next steps

If there are any follow-up meetings, make sure to include them in the summary and mentioned it specifically.

'''
)
refine_prompt_summary = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_summary,
)

# Initialize LLM
llm = ChatOpenAI(openai_api_key=openai_api_key,temperature=0.2, model_name="gpt-3.5-turbo-16k")

# Create a summary
sum_chain = load_summarize_chain(llm, chain_type="refine", verbose=True, question_prompt=PROMPT_SUMMARY, refine_prompt=refine_prompt_summary)
summary = sum_chain.run(docs)

# Write the response to a file
with open("summary.txt", "w") as f:
    f.write(summary)