from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI
import os
import dotenv

dotenv.load_dotenv()

chat_history = []

import re

def is_query_related_to_history(query, history):
    """
    Determine if the query is related to the recent history.
    """
    # Extract keywords from the recent history
    history_keywords = set(re.findall(r'\b\w+\b', history.lower()))
    
    # Check if any of the keywords appear in the query
    for keyword in history_keywords:
        if keyword in query.lower():
            return True
    return False

def get_recent_history_as_context(num_messages=5):
    """
    Get the last few messages (both user and bot) as a single string.
    """
    recent_history = chat_history[-num_messages:]
    return "\n".join([f"{sender if sender != 'bot' else ''}: {message}" for sender, message in recent_history])

def construct_index():
    # set maximum input size
    max_input_size = 4000
    # set number of output tokens
    num_outputs = 4000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    openai_api_key = os.getenv('OPENAI_API_KEY')

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    directory_path = "context_data/data"
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index


def convert_urls_to_links(text):
    # Find URLs in the text.
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    # Replace URLs with HTML links.
    return url_pattern.sub(lambda x: f'<a href="{x.group()}" target="_blank">{x.group()}</a>', text)

def get_response(query):
    # First, try to get an answer without any context
    if not os.path.exists('index.json'):
        construct_index()
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query("User: " + query)

    # If the response indicates uncertainty, use chat history as context
    uncertain_responses = [
        "I'm sorry, I'm not sure what your question or query is related to",
        "I don't know"
    ]
    if any(phrase in response.response for phrase in uncertain_responses):
        # Get the recent chat history as context
        context = get_recent_history_as_context()
        # If the query is related to history, combine the context with the current user query
        if is_query_related_to_history(query, context):
            full_query = context + "\nUser: " + query
            response = index.query(full_query)

    # Convert URLs in the response to clickable hyperlinks
    formatted_response = convert_urls_to_links(response.response)

    # Save the interaction to chat history
    chat_history.append(("user", query))
    chat_history.append(("bot", formatted_response))

    return formatted_response
