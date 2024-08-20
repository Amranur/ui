import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, Form
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import uuid

# Initialize FastAPI
app = FastAPI()

# Database setup
DATABASE_URL = "sqlite:///./sobjanta_sso.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# OAuth2 and Authentication models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    city = Column(String)
    hashed_password = Column(String)

class OAuth2Client(Base):
    __tablename__ = "oauth2_clients"
    client_id = Column(String, primary_key=True, index=True)
    client_secret = Column(String)
    redirect_uri = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship("User")

class OAuth2AuthorizationCode(Base):
    __tablename__ = "oauth2_auth_codes"
    code = Column(String, primary_key=True, index=True)
    client_id = Column(String, ForeignKey('oauth2_clients.client_id'))
    redirect_uri = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship("User")
    expires_at = Column(DateTime)

class OAuth2Token(Base):
    __tablename__ = "oauth2_tokens"
    access_token = Column(String, primary_key=True, index=True)
    client_id = Column(String, ForeignKey('oauth2_clients.client_id'))
    user_id = Column(Integer, ForeignKey('users.id'))
    expires_at = Column(DateTime)
    token_type = Column(String)

Base.metadata.create_all(bind=engine)

# Password encryption context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token handling
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

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

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# OAuth2 Password Bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

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

# Client Registration
@app.post("/register-client")
def register_client(redirect_uri: str, db: Session = Depends(get_db)):
    client_id = str(uuid.uuid4())
    client_secret = str(uuid.uuid4())
    client = OAuth2Client(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
    db.add(client)
    db.commit()
    return {"client_id": client_id, "client_secret": client_secret}

# Authorization Endpoint
@app.get("/authorize")
def authorize(response_type: str, client_id: str, redirect_uri: str, scope: str = None, state: str = None, db: Session = Depends(get_db)):
    client = db.query(OAuth2Client).filter(OAuth2Client.client_id == client_id).first()
    if not client or client.redirect_uri != redirect_uri:
        raise HTTPException(status_code=400, detail="Invalid client or redirect URI")

    # Simulate user login and consent
    user_id = 1  # Assume user is already logged in with ID 1
    auth_code = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(minutes=10)
    db.add(OAuth2AuthorizationCode(code=auth_code, client_id=client_id, redirect_uri=redirect_uri, user_id=user_id, expires_at=expires_at))
    db.commit()

    # Redirect back to the client with the authorization code
    redirect_response = f"{redirect_uri}?code={auth_code}"
    if state:
        redirect_response += f"&state={state}"
    return redirect_response

# Token Endpoint
@app.post("/token")
def issue_token(grant_type: str = Form(...), code: str = Form(...), redirect_uri: str = Form(...), client_id: str = Form(...), client_secret: str = Form(...), db: Session = Depends(get_db)):
    if grant_type != "authorization_code":
        raise HTTPException(status_code=400, detail="Unsupported grant type")

    client = db.query(OAuth2Client).filter(OAuth2Client.client_id == client_id, OAuth2Client.client_secret == client_secret).first()
    if not client:
        raise HTTPException(status_code=400, detail="Invalid client credentials")

    auth_code = db.query(OAuth2AuthorizationCode).filter(OAuth2AuthorizationCode.code == code, OAuth2AuthorizationCode.client_id == client_id, OAuth2AuthorizationCode.redirect_uri == redirect_uri).first()
    if not auth_code or auth_code.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired authorization code")

    access_token = create_access_token(data={"sub": str(auth_code.user_id)}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    db.add(OAuth2Token(access_token=access_token, client_id=client_id, user_id=auth_code.user_id, expires_at=datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES), token_type="bearer"))
    db.commit()
    return {"access_token": access_token, "token_type": "bearer"}

# User Data Endpoint
@app.get("/user-data")
def get_user_data(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    user = get_current_user(token, db)
    return {"name": user.name, "email": user.email, "city": user.city}

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
