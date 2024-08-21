import requests
import certifi
import re
from bs4 import BeautifulSoup
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.document_loaders import WebBaseLoader
from groq import Groq

# Initialize SearxSearchWrapper with your SearxNG host
search = SearxSearchWrapper(searx_host="http://127.0.0.1:32778")

def get_searxng_api_endpoint() -> str:
    return "http://localhost:32778"  # Replace with your actual SearxNG API endpoint

def search_searxng(query: str):
    searxng_url = get_searxng_api_endpoint()
    url = f"{searxng_url}/search?format=json"
    params = {'q': query}
    response = requests.get(url, params=params, verify=certifi.where())
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()

def clean_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

api_key = "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"

# Initialize Groq client with direct API key
client = Groq(api_key=api_key)

def summarize_content(content: str) -> str:
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Please summarize as an expert the following text: {content}",
                }
            ],
            model="llama-3.1-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return ''

# Main function to search and extract data
if __name__ == "__main__":
    query = "do python for loop code"
    results = search.results(query, num_results=10, engines=[])
    all_cleaned_content = []

    for result in results[:5]:
        url = result['link']
        print(f"Fetching {url}:")
        
        loader = WebBaseLoader(url)
        try:
            docs = loader.load()
            page_content = docs[0].page_content
            cleaned_content = clean_whitespace(page_content)
            all_cleaned_content.append(cleaned_content)
        except requests.exceptions.SSLError as e:
            print(f"SSL Error while fetching {url}: {e}")
        except Exception as e:
            print(f"Error while processing {url}: {e}")

    combined_content = "\n\n---\n\n".join(all_cleaned_content)
    summary = summarize_content(combined_content)
    
    print("Summary:")
    print(summary)




    