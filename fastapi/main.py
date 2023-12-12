from pydantic import BaseModel
from typing import Annotated, List
import pandas as pd
from scipy.spatial import distance
import ast
import openai
from transformers import GPT2TokenizerFast
import os
import asyncpg
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pinecone

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

SECRET_KEY = os.getenv('SECRET_KEY_VAR')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Define the GPT-3 model and other parameters
GPT_MODEL = "gpt-3.5-turbo"
api_key = os.environ.get('API_KEY')  # Replace with your actual OpenAI API key
openai.api_key = api_key

class User(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Database connection setup
async def connect_to_db():
    conn = await asyncpg.connect(
        user=os.getenv('USERNAME'),
        password=os.getenv('PASSWORD'),
        database=os.getenv('DBNAME'),
        host=os.getenv('ENDPOINT'),
        port=5432
    )
    print("Connection Setup Successful")
    return conn

# Create the table on startup
@app.on_event("startup")
async def startup_db():
    conn = await connect_to_db()
    await create_table(conn)

# Function to create the table if it doesn't exist
async def create_table(conn):
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS login_cred (
            username VARCHAR(50) PRIMARY KEY,
            password TEXT
        )
        """
    )


# Token generation and verification
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), conn = Depends(connect_to_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        user = await conn.fetchrow("SELECT * FROM login_cred WHERE username = $1", username)
        if user is None:
            raise credentials_exception
        
        # Retrieve the expiration time from the token's payload
        expiration_time = payload.get("exp")
        if expiration_time is None or datetime.utcfromtimestamp(expiration_time) < datetime.utcnow():
            raise credentials_exception  # Token has expired
        
        return User(username=user['username'], password=user['password'])
    except jwt.JWTError:
        raise credentials_exception
    
# Register endpoint
@app.post("/register")
async def register_user(form_data: OAuth2PasswordRequestForm = Depends(), conn = Depends(connect_to_db)):
    username = form_data.username
    password = form_data.password
    hashed_password = pwd_context.hash(password)
    query = "INSERT INTO login_cred (username, password) VALUES ($1, $2) ON CONFLICT DO NOTHING"
    await conn.execute(query, username, hashed_password)
    # access_token = create_access_token(data={"sub": username})
    # return {"access_token": access_token, "token_type": "Registered Successfully"}
    return {"message": "Signup successful"}


# Login endpoint
@app.post("/login", response_model=Token)
async def login_user(form_data: OAuth2PasswordRequestForm = Depends(), conn = Depends(connect_to_db)):
    username = form_data.username
    password = form_data.password
    user = await conn.fetchrow("SELECT * FROM login_cred WHERE username = $1", username)
    if user is None or not pwd_context.verify(password, user['password']):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}


############################################ Fetching Data from Pinecone for Streamlit Display ####################################
from fastapi import FastAPI
import os
import pinecone
from typing import List

class Query(BaseModel):  # Define a Pydantic model to properly parse the incoming JSON
    query: str

# Initialize Pinecone
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
env = "gcp-starter"
pinecone.init(api_key=PINECONE_API_KEY, environment=env)

# Create Pinecone index object
index_name = "openaiembeddings00"
index = pinecone.Index(index_name)


@app.get("/unique_pdf_names", response_model=List[str])
async def get_unique_pdf_names(token: str = Depends(oauth2_scheme), conn = Depends(connect_to_db)):
    stats = index.describe_index_stats()
    total_vector_count = stats['total_vector_count']
    
    unique_pdf_names = []
    seen = set()
    ids = [str(i) for i in range(0, total_vector_count)]

    # Fetch the data and get unique PDF names
    for vector_id in ids:
        response = index.fetch([vector_id])
        if vector_id in response['vectors']:
            metadata = response['vectors'][vector_id]['metadata']
            pdf_name = metadata.get('PDF_Name')
            if pdf_name and pdf_name not in seen:
                seen.add(pdf_name)
                unique_pdf_names.append(pdf_name)
    
    return unique_pdf_names

############################################################################ Extracting Context from PDF ###################

EMBEDDING_MODEL = "text-embedding-ada-002"  # Replace with your actual model

@app.post("/query_text")
async def query_text(query: Query, token: str = Depends(oauth2_scheme), conn = Depends(connect_to_db)):
    try:
        # Create the text embedding
        embedding_response = openai.Embedding.create(
            input=query.query,  # access 'query' field in Query model
            model=EMBEDDING_MODEL
        )
        embedding = embedding_response["data"][0]['embedding']

        # Query Pinecone with the generated embedding
        query_result = index.query(
            embedding,
            top_k=1,
            include_metadata = True, 
            get_score = True
        )
        
        first_match = query_result['matches'][0]
        return {
                'metadata': first_match['metadata'],
                'score': first_match['score']
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
########################################################################## Filtered Search #################################
class QueryFil(BaseModel):
    query: str
    pdf_name: str

@app.post("/query_text_filtered")
async def query_text_filter(data: QueryFil, token: str = Depends(oauth2_scheme), conn = Depends(connect_to_db)):
    try:
        # Create the text embedding
        embedding_response = openai.Embedding.create(
            input=data.query,  # Directly use data.query
            model=EMBEDDING_MODEL
        )
        embedding = embedding_response["data"][0]['embedding']

        # Query Pinecone with the generated embedding
        query_result = index.query(
            vector=embedding,  # Make sure this is the correct parameter name for your Pinecone client
            filter={"PDF_Name": {"$eq": data.pdf_name}},
            top_k=1,
            include_metadata=True,
            get_score=True
        )

        first_match = query_result['matches'][0]
    
    # Return only the metadata and score of the first match
        if first_match['score'] < 0.87:
            return {"message": "Solution not present in Selected Form"}
        else:
            # Return only the metadata and score of the first match
            return {
                'metadata': first_match['metadata'],
                'score': first_match['score']
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

###########################################################################################################################


