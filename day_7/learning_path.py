from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain import LLMChain
import os

# Set openai api key as environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Set OpenAI API key as environment variable
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# List of Youtube Urls
# If you want to load all the videos from a channel, use the code in youtube_ids.py
youtube_urls = [
    "https://www.youtube.com/watch?v=pEkxRQFNAs4", # Extract Topics From Video/Audio With LLMs (Topic Modeling w/ LangChain)
    "https://www.youtube.com/watch?v=GvQC5BHBkoM", # 8 Things New SaaS Developers Need To Know on DAY 1
    "https://www.youtube.com/watch?v=QoBCtcWO02g", # The One Person Business Model 2.0 (Turn Yourself Into A Business)
    "https://www.youtube.com/watch?v=mMv6OSuitWw", # Python 101: Learn the 5 Must-Know Concepts
    "https://www.youtube.com/watch?v=2zW5emKWof8", # 10 Years of Coding: What I Wish I Knew Before I Started
]

# Create a template for extracting topics from a text. 
extract_topics_template = """
    You are an expert in extracting skills being thaught from a transcript of a video.
    You're goal is to extract the skills thaught from the transcript below.
    The skills will be used to give the user an idea of what will be learned in the video.

    Transcript:
    ------------
    {text}
    ------------

    The description of the skills should be descriptive, but short and concise. Mention what overarching skill would be learned.
    
    Example:

    Implementing continuous delivery for faster shipping - Software development
    Evaluating and selecting a suitable tech stack for SaaS development - Software development
    Recognizing the importance of marketing and customer communication in building a successful SaaS business - Business and marketing

    Don't add numbers. Just each skill on a new line.

    SKILLS - OVERARCHING SKILL:
"""

PROMPT_EXTRACT_TOPICS = PromptTemplate(template=extract_topics_template, input_variables=["text"])


# The second prompt is for the refinement of the summary and topics, based on subsequent chunks.
extract_topics_refine_template = (
"""
    You are an expert in extracting skills from a transcript of a video.
    You're goal is to extract the skills thaught from the transcript below.
    The skills will be used to give the user an idea of what will be learned in the video.

    We have provided a list of skills up to a certain point: {existing_answer}
    We have the opportunity to refine the skills
    (only if needed) with some more context below.
    ------------
    {text}
    ------------
    Given the new context, refine the skills discussed.
    If the context isn't useful, return the list of skills.
    The description of the skills should be descriptive, but short and concise. Mention what overarching skill would be learned.

    Example:

    Implementing continuous delivery for faster shipping - Software development
    Evaluating and selecting a suitable tech stack for SaaS development - Software development
    Recognizing the importance of marketing and customer communication in building a successful SaaS business - Business and marketing

    Don't add numbers. Just each skill on a new line.

    SKILLS - OVERARCHING SKILL:
"""
)

PROMPT_EXTRACT_TOPICS_REFINE = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=extract_topics_refine_template,
)

# Prompt for genarting a list of subskills needed to master a skill

subskills_template = """
You are an assistant specialized in desiging learning paths for people trying to acquire a particular skill-set. 

Your goal is to make a list of sub skills a person needs to become proficient in a particular skill.

The skill set you need to design a learning path for is: {skill_set}

The user will say which skill set they want to learn, and you'll provide a short and consice list of specific skills this person needs to learn. 

This list will be used to find YouTube videos related to those skills. Don't mention youtube videos though! Name only 5 skills maximum.
"""

PROMPT_SUBSKILLS = PromptTemplate(template=subskills_template, input_variables=["skill_set"])

# Prompt for finding a video based on a subskill set

find_video_template = """
You are an assistant specialized in desiging learning paths for people trying to acquire a particular skill-set.

Your goal is to find a list of videos that teaches a particular skill.

It should be based on the following context:

{context}

Look for videos that teach the following skills: {skill_set}

RETURN A LIST OF VIDEOS WITH YOUTUBE URL AND TITLE:
"""

PROMPT_FIND_VIDEO = PromptTemplate(template=find_video_template, input_variables=["context","skill_set"])

# Initialize the large language model. You can use the gpt-3.5-turbo-16k model or any model you prefer.
# Play around with the temperature parameter to get different results. Higher temperature means more randomness. Lower temperature means more deterministic.
llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0)


# Initialize empty document list
documents = []

with get_openai_callback() as cb:

    # Loop over the youtube urls
    for url in youtube_urls:

        # Load a youtube video and get the transcript
        youtube_url = url
        loader = YoutubeLoader.from_youtube_url(youtube_url=youtube_url, add_video_info=True)
        data = loader.load()
        metadata = data[0].metadata
        title = metadata['title']
        author = metadata['author']

        # Split the transcript into shorter chunks.
        # First create the text splitter. The chunk_size is the maximum number of tokens in each chunk.
        text_splitter = TokenTextSplitter(chunk_size = 2000, chunk_overlap = 100)

        # Then split the transcript into chunks.
        # The .split_documents() method returns the page_content attribute of the Document object.
        docs = text_splitter.split_documents(data)

        # Initialize the summarization chain
        extract_topics_chain = load_summarize_chain(llm=llm, chain_type="refine", verbose=True, question_prompt = PROMPT_EXTRACT_TOPICS, refine_prompt = PROMPT_EXTRACT_TOPICS_REFINE)
        extracted_topics = extract_topics_chain(docs)

        video_overview = ""

        # Add the YouTube Channel name, video title, URL and extracted topics to the video overview
        video_overview += f"YouTube Channel: {author}\n"
        video_overview += f"YouTube Video: {title}\n"
        video_overview += f"YouTube URL: {youtube_url}\n"
        video_overview += "Skills: \n"
        video_overview += extracted_topics['output_text']

        # Create a document object with the video overview
        docs = Document(page_content=video_overview, metadata={"title": title, "author": author, "url": youtube_url})

        # Add the document to the documents list
        documents.append(docs)

    # Initialize the embeddings
    embeddings = OpenAIEmbeddings()

    # Create a vector store with the documents  
    vector_store = FAISS.from_documents(documents, embeddings)

    # Save the vector store
    vector_store.save_local("vector/", "vector_store")

    # Load the vector store
    vector_store = FAISS.load_local("vector/", embeddings, "vector_store")

    # Initialize the subskills chain
    subskills_chain = LLMChain(llm=llm, prompt=PROMPT_SUBSKILLS)

    # Loop for questions
    while True:
        # Ask the user what skill they want to learn
        skill_set = input("What skill set do you want to learn? ")

        # Use skillset to find subskills
        subskills = subskills_chain.predict(skill_set=skill_set)

        # Print subskills
        print(f"Subskills: \n {subskills}\n")

        # Initialize the retrieval chain
        qa = RetrievalQA.from_chain_type(llm = llm, retriever = vector_store.as_retriever(), chain_type="stuff", verbose=True)

        # Set query to ask
        query = f"Which Youtube videos teach {subskills}?"

        # Use query to find videos
        videos = qa.run(query)

        # Print videos
        print(f"Videos: \n {videos}\n")
    