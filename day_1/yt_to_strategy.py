"""
This script shows how to create a strategy for a four-hour workday based on a YouTube video.
We're using an easy LangChain implementation to show how to use the different components of LangChain.
This is part of my '7 Days of LangChain' series. 

Check out the explanation about the code on my Twitter (@JorisTechTalk)

"""


from langchain import LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:

    # Set your OpenAI API Key.
    openai_api_key = 'YOUR_API_KEY_HERE'

    # Load a youtube video and get the transcript
    url = "https://www.youtube.com/watch?v=aV4jKPFOjvk"
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    data = loader.load()

    # Split the transcript into shorter chunks.
    # First create the text splitter. The chunk_size is the maximum number of tokens in each chunk.
    # With the new gpt-3.5-turbo-16k model, you actually don't need it in this example, but it's good to know how to do it.
    text_splitter = TokenTextSplitter(chunk_size = 5000, chunk_overlap = 100)

    # Then split the transcript into chunks.
    # The .split_documents() method returns the page_content attribute of the Document object.
    docs = text_splitter.split_documents(data)

    # Create the prompts. The prompt is the instruction to the model. Prompting is key to getting good results.
    # Play around with the prompt to get different results.
    # We create two prompts. Since we will be using the refine summarize chain, we need a prompt for the initial 'summarization' of a chunk, and a prompt for the refinement of the summary of subsequent chunks.


    # The first prompt is for the initial summarization of a chunk. You can add any info about yourself or the topic you want.
    # You could specifically focus on a skill you have to get more relevant results.
    strategy_template = """
        You are an expert in creating strategies for getting a four-hour workday. You are a productivity coach and you have helped many people achieve a four-hour workday.
        You're goal is to create a detailed strategy for getting a four-hour workday.
        The strategy should be based on the following text:
        ------------
        {text}
        ------------
        Given the text, create a detailed strategy. The strategy is aimed to get a working plan on how to achieve a four-hour workday.
        The strategy should be as detailed as possible.
        STRATEGY:
    """

    PROMPT_STRATEGY = PromptTemplate(template=strategy_template, input_variables=["text"])


    # The second prompt is for the refinement of the summary, based on subsequent chunks.
    strategy_refine_template = (
    """
        You are an expert in creating strategies for getting a four-hour workday.
        You're goal is to create a detailed strategy for getting a four-hour workday.
        We have provided an existing strategy up to a certain point: {existing_answer}
        We have the opportunity to refine the strategy
        (only if needed) with some more context below.
        ------------
        {text}
        ------------
        Given the new context, refine the strategy.
        The strategy is aimed to get a working plan on how to achieve a four-hour workday.
        If the context isn't useful, return the original strategy.
    """
    )

    PROMPT_STRATEGY_REFINE = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=strategy_refine_template,
    )

    # Initialize the large language model. You can use the gpt-3.5-turbo-16k model or any model you prefer.
    # Play around with the temperature parameter to get different results. Higher temperature means more randomness. Lower temperature means more deterministic.
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo-16k', temperature=0.5)

    # Initiliaze the chain.
    # The verbose parameter prints the 'thought process' of the model. It's useful for debugging.
    strategy_chain = load_summarize_chain(llm=llm, chain_type='refine', verbose=True, question_prompt=PROMPT_STRATEGY, refine_prompt=PROMPT_STRATEGY_REFINE)
    strategy = strategy_chain.run(docs)

    # Now write the strategy to a file.
    with open('strategy.txt', 'w') as f:
        f.write(strategy)

    # Now use this strategy to create a plan.
    # The plan is a list of steps to take to achieve the goal.
    # The plan is based on the strategy.

    # Create the prompt for the plan.
    plan_template = """
        You are an expert in creating plans for getting a four-hour workday. You are a productivity coach and you have helped many people achieve a four-hour workday.
        You're goal is to create a detailed plan for getting a four-hour workday.
        The plan should be based on the following strategy:
        ------------
        {strategy}
        ------------
        Given the strategy, create a detailed plan. The plan is aimed to get a working plan on how to achieve a four-hour workday.
        Think step by step.
        The plan should be as detailed as possible.
        PLAN:
    """

    PROMPT_PLAN = PromptTemplate(template=plan_template, input_variables=["strategy"])

    # Initialize the chain.
    plan_chain = LLMChain(llm=llm, prompt=PROMPT_PLAN, verbose=True)
    plan = plan_chain(strategy)

    # Now write the plan to a file.
    with open('plan.txt', 'w') as f:
        f.write(plan['text'])

# Print the total cost of the API calls.
print(cb)