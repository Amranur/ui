from typing import List, Optional
from schemas import APIDocumentationCreate, APIDocumentationUpdate
import uvicorn
from fastapi import FastAPI, HTTPException, Depends,status
from fastapi import BackgroundTasks
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
from sqlalchemy import String, Column
from sqlalchemy import Text


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
DATABASE_URL = "mysql+mysqlconnector://root@localhost:3306/sobjanta_api"  # Replace with your MySQL credentials
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))  # Specify length for MySQL compatibility
    email = Column(String(255), unique=True, index=True)  # Specify length
    city = Column(String(255), nullable=True)  # Specify length, nullable if not always present
    hashed_password = Column(String(255))
    role = Column(String(50), default="customer")  # Specify length
    email_verified = Column(Boolean, default=False)
    ev_code = Column(String(6), nullable=True)  # Assuming a 6-character code
    ev_code_expire = Column(DateTime, nullable=True)

class APIKey(Base):
    __tablename__ = "api_keys"
    
    key = Column(String(50), primary_key=True, index=True)  # Specify length
    name = Column(String(255), nullable=False)  # Specify length
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(Boolean, default=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)  # Ensure ForeignKey is not nullable
    user = relationship("User")

class RequestLog(Base):
    __tablename__ = "request_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String(50), index=True)  # Specify length
    query = Column(String(1024), nullable=True)  # Larger length for query logs, nullable if not always present
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    
class API_Documentation(Base):
    __tablename__ = "api_documentation"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    section = Column(String(50), nullable=False)
    content = Column(Text(), nullable=False)
    example_code = Column(Text(), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


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
def register_customer(
    data: UserCreateRequest, 
    db: Session = Depends(get_db),
    background_tasks=BackgroundTasks
):
    hashed_password = get_password_hash(data.password)
    ev_code, ev_code_expire = generate_ev_code_and_expiry()

    user = User(
        name=data.name,
        email=data.email,
        city=data.city,
        hashed_password=hashed_password,
        role="customer",
        email_verified=False,
        ev_code=ev_code,
        ev_code_expire=ev_code_expire
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Send verification email in the background
   # background_tasks.add_task(send_verification_email, user.email, ev_code)

    return {"message": "Customer registered successfully. Please check your email for verification."}

def generate_six_digit_code():
    return str(random.randint(100000, 999999))

def generate_ev_code_and_expiry():
    code = generate_six_digit_code()
    expiry = datetime.utcnow() + timedelta(minutes=5)
    return code, expiry

def send_verification_email(email: str, ev_code: str):
    print("come1")
    # SMTP server configuration
    smtp_host = "mail.sobjanta.ai"
    smtp_port = 465  # SSL/TLS port
    smtp_user = "sobjanta@sobjanta.ai"
    smtp_password = "!ry1wrI_cV@$"
    print("come12")
    # Email content
    subject = "Your Email Verification Code"
    body = f"Your verification code is {ev_code}. It will expire in 5 minutes."

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    try:
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()  # Secure the connection
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, email, msg.as_string())
        server.quit()
        print("Verification email sent successfully")
    except Exception as e:
        print(f"Failed to send verification email: {e}")


@app.post("/verify-email")
def verify_email(gmail: str, ev_code: str, db: Session = Depends(get_db)):
    # Find the user by their email
    user = db.query(User).filter(User.email == gmail).first()

    if not user:
        raise HTTPException(status_code=400, detail="User with this email does not exist")

    if user.ev_code != ev_code:
        raise HTTPException(status_code=400, detail="Invalid verification code")

    if user.ev_code_expire < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Verification code has expired")

    # If everything is okay, verify the email
    user.email_verified = True
    user.ev_code = None  # Optionally clear the verification code
    user.ev_code_expire = None  # Clear the expiration time
    db.commit()
    access_token = create_access_token(data={"sub": str(user.id)}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"message": "Email verified successfully","acess_token":access_token}

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

@app.post("/login")
def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": str(user.id)}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "Bearer"}

@app.post("/login2")
def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db),
    background_tasks=BackgroundTasks
):
    user = db.query(User).filter(User.email == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    if not user.email_verified:
        # Generate a new verification code and update the expiration
        ev_code, ev_code_expire = generate_ev_code_and_expiry()
        user.ev_code = ev_code
        user.ev_code_expire = ev_code_expire
        db.commit()

        # Send the verification email
        #background_tasks.add_task(send_verification_email, user.email, user.ev_code)

        raise HTTPException(status_code=400, detail="Email not verified. A new verification code has been sent to your email.")

    # If email is verified, create the access token
    access_token = create_access_token(
        data={"sub": str(user.id)}, 
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
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

##api doc
def create_documentation(db: Session, doc: APIDocumentationCreate):
    db_doc = API_Documentation(
        title=doc.title,
        section=doc.section,
        content=doc.content,
        example_code=doc.example_code
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc

def get_documentation(db: Session, doc_id: int):
    return db.query(API_Documentation).filter(API_Documentation.id == doc_id).first()

def get_all_documentation(db: Session, skip: int = 0, limit: int = 10, 
                           title: Optional[str] = None, section: Optional[str] = None, 
                           sort_field: Optional[str] = None, sort_order: Optional[str] = None):
    query = db.query(API_Documentation)
    
    if title:
        query = query.filter(API_Documentation.title.like(f"%{title}%"))
    
    if section:
        query = query.filter(API_Documentation.section.like(f"%{section}%"))
    
    if sort_field and sort_order:
        if sort_order.lower() == "asc":
            query = query.order_by(getattr(API_Documentation, sort_field).asc())
        elif sort_order.lower() == "desc":
            query = query.order_by(getattr(API_Documentation, sort_field).desc())
        else:
            raise HTTPException(status_code=400, detail="Invalid sort order. Use 'asc' or 'desc'.")
    
    # Apply offset and limit after filtering and sorting
    query = query.offset(skip).limit(limit)
    
    return query.all()

def update_documentation(db: Session, doc_id: int, doc_update: APIDocumentationCreate):
    db_doc = db.query(API_Documentation).filter(API_Documentation.id == doc_id).first()
    if db_doc:
        db_doc.title = doc_update.title
        db_doc.section = doc_update.section
        db_doc.content = doc_update.content
        db_doc.example_code = doc_update.example_code
        db.commit()
        db.refresh(db_doc)
        return db_doc
    return None

def delete_documentation(db: Session, doc_id: int):
    db_doc = db.query(API_Documentation).filter(API_Documentation.id == doc_id).first()
    if db_doc:
        db.delete(db_doc)
        db.commit()
        return db_doc
    return None

@app.post("/documentation/")
def create_documentation_endpoint(doc: APIDocumentationCreate, db: Session = Depends(get_db)):
    return create_documentation(db=db, doc=doc)

@app.get("/documentation/{doc_id}")
def read_documentation(doc_id: int, db: Session = Depends(get_db)):
    db_doc = get_documentation(db, doc_id)
    if db_doc is None:
        raise HTTPException(status_code=404, detail="Documentation not found")
    return db_doc

@app.get("/documentation/")
def read_all_documentation(skip: int = 0, limit: int = 10, 
                           title: Optional[str] = None, section: Optional[str] = None, 
                           sort_field: Optional[str] = None, sort_order: Optional[str] = None,
                           db: Session = Depends(get_db)):
    return get_all_documentation(db, skip, limit, title, section, sort_field, sort_order)

@app.put("/documentation/{doc_id}")
def update_documentation_endpoint(doc_id: int, doc: APIDocumentationUpdate, db: Session = Depends(get_db)):
    updated_doc = update_documentation(db=db, doc_id=doc_id, doc_update=doc)
    if updated_doc is None:
        raise HTTPException(status_code=404, detail="Documentation not found")
    return updated_doc

@app.delete("/documentation/{doc_id}")
def delete_documentation_endpoint(doc_id: int, db: Session = Depends(get_db)):
    deleted_doc = delete_documentation(db, doc_id)
    if deleted_doc is None:
        raise HTTPException(status_code=404, detail="Documentation not found")
    return deleted_doc


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
