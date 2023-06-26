# PODCAST Q&A BOT

from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

with get_openai_callback() as cb:

    # Load a youtube video and get the transcript
    loader = YoutubeLoader.from_youtube_url('https://www.youtube.com/watch?v=-hxeDjAxvJ8', add_video_info=True)
    data = loader.load()

    # Initialize text splitter for summary (Large chunks for better context and less API calls)
    text_splitter_summary = TokenTextSplitter(chunk_size = 10000, chunk_overlap = 250)

    # Split text into docs for summary
    docs_summary = text_splitter_summary.split_documents(data)

    # Initialize text splitter for QA (Smaller chunks for better QA)
    text_splitter_qa = TokenTextSplitter(chunk_size = 1000, chunk_overlap = 200)

    # Split text into docs for QA
    docs_qa = text_splitter_qa.split_documents(data)

    # Prompts for summary

    # The first prompt is for the initial summarization of a chunk. You can add any info about yourself or the topic you want.
    # You could specifically focus on a skill you have to get more relevant results.
    summary_template = """
        You are an expert in summarizing YouTube videos.
        You're goal is to create a summary of a podcast.
        Below you find the transcript of a podcast:
        ------------
        {text}
        ------------

        The transript of the podcast will also be used as the basis for a question and answer bot.
        Provide some examples questions and answers that could be asked about the podcast. Make these questions very specific.

        Total output will be a summary of the video and a list of example questions the user could ask of the video.

        SUMMARY AND QUESTIONS:
    """

    PROMPT_SUMMARY = PromptTemplate(template=summary_template, input_variables=["text"])


    # The second prompt is for the refinement of the summary, based on subsequent chunks.
    summary_refine_template = (
    """
        You are an expert in summarizing YouTube videos.
        You're goal is to create a summary of a podcast.
        We have provided an existing summary up to a certain point: {existing_answer}
        We have the opportunity to refine the summary
        (only if needed) with some more context below.
        Below you find the transcript of a podcast:
        ------------
        {text}
        ------------
        Given the new context, refine the summary and example questions.
        The transript of the podcast will also be used as the basis for a question and answer bot.
        Provide some examples questions and answers that could be asked about the podcast. Make these questions very specific.
        If the context isn't useful, return the original summary and questions.
        Total output will be a summary of the video and a list of example questions the user could ask of the video.

        SUMMARY AND QUESTIONS:
    """
    )

    PROMPT_SUMMARY_REFINE = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=summary_refine_template,
    )

    # Set OPENAI API key
    openai_api_key = 'YOUR_API_KEY'

    # Initialize LLM
    llm_summary = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo-16k', temperature=0.3)

    # Initialize summarization chain
    summarize_chain = load_summarize_chain(llm=llm_summary, chain_type="refine", verbose=True, question_prompt=PROMPT_SUMMARY, refine_prompt=PROMPT_SUMMARY_REFINE)
    summary = summarize_chain.run(docs_summary)

    # Write summary to file
    with open("summary.txt", "w") as f:
        f.write(summary)

    # Create the LLM model for the question answering
    llm_question_answer = ChatOpenAI(openai_api_key=openai_api_key,temperature=0.2, model="gpt-3.5-turbo-16k")

    # Create the vector database and RetrievalQA Chain
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs_qa, embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm_question_answer, chain_type="stuff", retriever=db.as_retriever())


    question = ""
    # Run the QA chain continuously
    while question != "exit":
        # Get the user question
        question = input("Ask a question or enter exit to close the app: ")
        # Run the QA chain
        answer = qa.run(question)
        print(answer)
        print("---------------------------------")
        print("\n")

print(cb)