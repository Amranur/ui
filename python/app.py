from typing import List
import uvicorn
from fastapi import FastAPI, HTTPException, Depends,status
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import uuid
import requests
import certifi
import re
from bs4 import BeautifulSoup
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.document_loaders import WebBaseLoader
from groq import Groq
from fastapi.security import OAuth2PasswordBearer
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr


# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, or specify a list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Database setup
DATABASE_URL = "sqlite:///./info_api.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    city = Column(String)
    hashed_password = Column(String)
    role = Column(String, default="customer")

class APIKey(Base):
    __tablename__ = "api_keys"
    
    key = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)  # Add the name field
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(Boolean, default=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship("User")


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

# Password encryption context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token handling
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 3000

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta if expires_delta else datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
    #"sobjanta-" +

# Define OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise credentials_exception
        return user
    except JWTError:
        raise credentials_exception

def role_required(roles: List[str]):
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return current_user
    return role_checker


class UserCreateRequest(BaseModel):
    name: str
    email: str
    password: str
    city: str

# Register as Customer
@app.post("/register-customer")
def register_customer(data: UserCreateRequest, db: Session = Depends(get_db)):
    hashed_password = get_password_hash(data.password)
    user = User(
        name=data.name,
        email=data.email,
        city=data.city,
        hashed_password=hashed_password,
        role="customer"  # Fixed role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "Customer registered successfully"}

# Register as Admin
@app.post("/register-admin")
def register_admin(data: UserCreateRequest, db: Session = Depends(get_db)):
    hashed_password = get_password_hash(data.password)
    user = User(
        name=data.name,
        email=data.email,
        city=data.city,
        hashed_password=hashed_password,
        role="admin"  # Fixed role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "Admin registered successfully"}

# User login and token generation

# @app.post("/login")
# def login_user(email: str, password: str, db: Session = Depends(get_db)):
#     user = db.query(User).filter(User.email == email).first()
#     if not user or not verify_password(password, user.hashed_password):
#         raise HTTPException(status_code=400, detail="Invalid credentials")
#     access_token = create_access_token(data={"sub": str(user.id)}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
#     return {"access_token": access_token, "token_type": "sobjanta"}

# Login endpoint for obtaining a token

@app.post("/login")
def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": str(user.id)}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "Bearer"}

# API Key generation

class APIKeyCreateRequest(BaseModel):
    name: str

@app.post("/generate-api-key")
def generate_api_key(
    api_key_data: APIKeyCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if an API key with the same name already exists for the current user
    existing_key = db.query(APIKey).filter(
        APIKey.name == api_key_data.name,
        APIKey.user_id == current_user.id
    ).first()

    if existing_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="API key name must be unique for the current user")

    # Generate a new API key
    key = str(uuid.uuid4())
    db_key = APIKey(key=key, name=api_key_data.name, user_id=current_user.id)
    db.add(db_key)
    db.commit()
    db.refresh(db_key)
    
    return {"api_key": db_key.key, "status": "active", "name": db_key.name}
# Disable an API key
@app.post("/disable-api-key")
def disable_api_key(api_key: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_key = db.query(APIKey).filter(APIKey.key == api_key, APIKey.user_id == current_user.id).first()
    if not db_key:
        raise HTTPException(status_code=404, detail="API key not found")
    db_key.status = False
    db.commit()
    return {"message": "API key disabled"}


# Get all API keys for the current user
@app.get("/api-keys")
def get_api_keys(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    keys = db.query(APIKey).filter(APIKey.user_id == current_user.id).all()
    return {"api_keys": [{"key": key.key, "name": key.name,"created_at": key.created_at, "status": "active" if key.status else "disabled"} for key in keys]}

# Get all API keys
@app.get("/api-keys-all")
def get_api_keys(db: Session = Depends(get_db)):
    keys = db.query(APIKey).all()
    return {"api_keys": [{"key": key.key, "created_at": key.created_at, "status": "active" if key.status else "disabled"} for key in keys]}

@app.get("/request-logs-all")
def get_all_request_logs(db: Session = Depends(get_db)):
    # Retrieve all request logs
    logs = db.query(RequestLog).all()
    
    # Format the logs for response
    return {"request_logs": [
        {"id": log.id, "api_key": log.api_key, "query": log.query, "timestamp": log.timestamp} 
        for log in logs
    ]}

# Get all request logs for the current user's API keys
@app.get("/request-logs-current-user")
def get_request_logs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    keys = db.query(APIKey).filter(APIKey.user_id == current_user.id).all()
    api_keys = [key.key for key in keys]
    logs = db.query(RequestLog).filter(RequestLog.api_key.in_(api_keys)).all()
    return {"request_logs": [{"id": log.id, "api_key": log.api_key, "query": log.query, "timestamp": log.timestamp} for log in logs]}

# Get all request logs by api key
@app.get("/request-logs")
def get_request_logs(api_key: str, current_user: User = Depends(get_current_user),db: Session = Depends(get_db)):
    # Check if API key exists
    db_key = db.query(APIKey).filter(APIKey.key == api_key).first()
    if not db_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    logs = db.query(RequestLog).filter(RequestLog.api_key == api_key).all()
    return {"total_logs": len(logs),"request_logs": [{"id": log.id, "query": log.query, "timestamp": log.timestamp} for log in logs]}

@app.post("/admin-only")
def admin_only_endpoint(current_user: User = Depends(role_required(["admin"]))):
    return {"message": "This is an admin-only endpoint"}

# Clean whitespace function
def clean_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Initialize Groq client with direct API key
api_key = "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"
client = Groq(api_key=api_key)

# Search function using the API key
@app.get("/search")
def search_searxng(query: str, api_key: str, db: Session = Depends(get_db)):
    # Check if API key exists and is active
    db_key = db.query(APIKey).filter(APIKey.key == api_key, APIKey.status == True).first()
    if not db_key:
        raise HTTPException(status_code=403, detail="Invalid or disabled API key")

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
    uvicorn.run(app, host="127.0.0.1", port=8000)
