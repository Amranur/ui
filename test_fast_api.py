import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import uuid
import requests
import certifi
import re
from bs4 import BeautifulSoup
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.document_loaders import WebBaseLoader
from groq import Groq

# Initialize FastAPI
app = FastAPI()

# Database setup
DATABASE_URL = "sqlite:///./search_api.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class APIKey(Base):
    __tablename__ = "api_keys"
    key = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class RequestLog(Base):
    __tablename__ = "request_logs"
    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String, index=True)
    query = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create the database tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API Key generation
@app.post("/generate-api-key")
def generate_api_key(db: Session = Depends(get_db)):
    key = str(uuid.uuid4())
    db_key = APIKey(key=key)
    db.add(db_key)
    db.commit()
    return {"api_key": key}

# Search function using the API key
@app.get("/search")
def search_searxng(query: str, api_key: str, db: Session = Depends(get_db)):
    # Check if API key exists
    db_key = db.query(APIKey).filter(APIKey.key == api_key).first()
    if not db_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Log the request
    log = RequestLog(api_key=api_key, query=query)
    db.add(log)
    db.commit()

    # Perform the search using SearxNG
    search = SearxSearchWrapper(searx_host="http://127.0.0.1:32778")
    results = search.results(query, num_results=10, engines=[])
    all_cleaned_content = []

    for result in results[:10]:
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
    
    return {"summary": summary}

# Get all API keys
@app.get("/api-keys")
def get_api_keys(db: Session = Depends(get_db)):
    keys = db.query(APIKey).all()
    return {"api_keys": [{"key": key.key, "created_at": key.created_at} for key in keys]}

# Get all request logs
@app.get("/request-logs")
def get_request_logs(api_key: str, db: Session = Depends(get_db)):
    # Check if API key exists
    db_key = db.query(APIKey).filter(APIKey.key == api_key).first()
    if not db_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    logs = db.query(RequestLog).filter(RequestLog.api_key == api_key).all()
    return {"request_logs": [{"id": log.id, "query": log.query, "timestamp": log.timestamp} for log in logs]}


# Clean whitespace function
def clean_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Initialize Groq client with direct API key
api_key = "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"
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

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
