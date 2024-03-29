import time 
import os 

from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
import jwt
# The code is importing necessary modules and libraries for the application. Here's a
# breakdown of what each import statement does:

import sys


from fastapi import FastAPI, Depends, HTTPException, status, Header, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, OAuth2PasswordBearer
from fastapi.security import OAuth2PasswordRequestForm, HTTPAuthorizationCredentials    
from fastapi.responses import FileResponse 
from datetime import datetime, timedelta
from passlib.context import CryptContext
from typing import Optional, Annotated
from pydantic import BaseModel
import dotenv 

from loguru import logger 
from dataputket_rpg import initialize, RPG  

dotenv.load_dotenv()

app = FastAPI()
security = HTTPBearer()

SECRET_KEY = os.getenv("SECRET_KEY")
SECRET_KEY ="your_secret_key"
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") 
hash1 = pwd_context.hash("dataputket_test_Lollero123")
USERS = [
    {"username": "dataputket_test", "password": hash1},
]

access_token = None 
  
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)      

def authenticate_token(token: OAuth2PasswordBearer = Depends(oauth2_scheme)):
    
    print(f"token: {token}")
    print(f"access_token: {os.getenv('ACCESS_TOKEN')}")
    
    access_token = os.getenv("ACCESS_TOKEN")  
    
    if token != access_token:
        raise HTTPException(status_code=401, 
                            detail="Invalid authentication credentials", 
                            headers={"WWW-Authenticate": "Bearer"},
                            )
    return True 
   
def authenticate_user(username: str, password: str):
    user = next(
        (user for user in USERS if user["username"] == username), None
    )
    if not user:
        return False
    if not verify_password(password, user["password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    #encoded_jwt = jwt.encode_payload(to_encode)
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"}) 
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    user = next((user for user in USERS if user["username"] == username), None)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found", headers={"WWW-Authenticate": "Bearer"})
    return user


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") 
hash1 = pwd_context.hash("dataputket_test_Lollero123")
USERS = [
    {"username": "dataputket_test", "password": hash1},
]

access_token = None 


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)      

def authenticate_token(token: OAuth2PasswordBearer = Depends(oauth2_scheme)):
    
    print(f"token: {token}")
    print(f"access_token: {os.getenv('ACCESS_TOKEN')}")
    
    access_token = os.getenv("ACCESS_TOKEN")  
    
    if token != access_token:
        raise HTTPException(status_code=401, 
                            detail="Invalid authentication credentials", 
                            headers={"WWW-Authenticate": "Bearer"},
                            )
    return True 
   
def authenticate_user(username: str, password: str):
    user = next(
        (user for user in USERS if user["username"] == username), None
    )
    if not user:
        return False
    if not verify_password(password, user["password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    #encoded_jwt = jwt.encode_payload(to_encode)
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    user = next((user for user in USERS if user["username"] == username), None)
    if user is None:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="User not found", headers={"WWW-Authenticate": "Bearer"})
    return user 

# Fastapi expects this post data from client
# The `AppParams` class represents the parameters for generating an image, including the
# user prompt, version number, number of steps, model name, activation status, base prompt
# usage, base prompt ratio, batch size, random seed, configuration, height, and width of
# the image.
class AppParams(BaseModel):
    # User prompt: Text prompt from client to use as base to describe image content
    user_prompt: str = "Yellow Maserati sport car on Alp mountain highway in the morning"
    # Version number: 0 = multi-attribute, 1 = complex object, 2 = natural object, 3 = human-like
    version_number: int = 0
    # Steps: Number of iterations for image generation
    steps: int = 20
    # model_name: Name of model to use for image generation
    # (albedobaseXL_v20.safetensors is predownloaded to server) 
    model_name: str = 'albedobaseXL_v20.safetensors'
    # activate: Activate regional settings (0 = deactivate, 1 = activate)
    activate: int = 1
    # use_base: Use base prompt (0 = deactivate, 1 = activate)
    use_base: int = 0
    # base_ratio: Ratio of base prompt to regional prompt
    base_ratio: float = 0.3
    # base_prompt: Base prompt to use as base to describe image content
    base_prompt: str = ""
    # batch_size: Batch size for image generation
    batch_size: int = 1
    # seed: Random seed for image generation
    seed: int = 1234
    # cfg: Configuration for image generation
    cfg: int = 5
    # height: Height of image to generate
    height: int = 1024
    # width: Width of image to generate
    width: int = 1024


# Finally create application, add router and run app
app = FastAPI()

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token({"sub": user["username"]})
    os.environ["ACCESS_TOKEN"] = access_token   


@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user  

@app.post("/dataputket_rpg")
async def rpg(params: AppParams, Authorization: Optional[str] = Header(None)):
    # Get header from request
    # Get request
    #headers = request.headers
    #token = headers.get("Authorization")
    print(Authorization)
    token = Authorization.split("Bearer ")[1]
    token =  jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    print(params)

    version_list = ["multi-attribute","complex-object"]
    logger.debug(f"params: {params}")
    user_prompt = params.user_prompt
    version_number = params.version_number
    steps = params.steps
    model_name = params.model_name
    activate = params.activate
    use_base = params.use_base
    base_ratio = params.base_ratio
    base_prompt = params.base_prompt
    batch_size = params.batch_size
    seed = params.seed
    cfg = params.cfg
    steps = params.steps
    height = params.height
    width = params.width

    if version_number >= len(version_list):
        logger.error(f"version_number {version_number} is out of range")
        return {"error": "version_number is out of range"}

    if version_number < 0:
        logger.error(f"version_number {version_number} is out of range")
        return {"error": "version_number is out of range"}

    if version_number >= 0:
        version = version_list[version_number]
        
    appendix = 'gpt4'
    
    if 1 == activate:
        activate = True 
    else:
        activate = False 
        
    if 1 == use_base:
        use_base = True
    else:
        use_base = False

    # Initialize model
    initialize(model_name= 'albedobaseXL_v20.safetensors')
     
       
    image=RPG(user_prompt=user_prompt,
        diffusion_model=model_name,
        version=version,
        split_ratio=None,
        use_base=use_base,
        base_ratio=base_ratio,
        base_prompt=base_prompt,
        batch_size=batch_size,
        seed=seed,
        use_personalized=False,
        cfg=cfg,
        steps=steps,
        height=height,
        width=width)
    
    logger.debug(f"len images: {len(image)}")
    file_names = []
    
    # Save images to server (TODO: save to gcp bucket for persistence)
    for i in range(len(image)):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"{appendix}_image_{timestamp}.png"
        file_names.append(file_name)
        image[i].save(f"generated_imgs/{file_name}")

    # Create response and send it 
    # At the moment you can use FastAPI endpoint "/docs/" on your browser to create post
    # call with json-data for parameters and view image result after a while 
    # (processing can take a while)
    # TODO: After images are saved to bucket, maybe create create get request to 
    # retrieve latest image from bucket
    return FileResponse(f"generated_imgs/{file_names[0]}", media_type="image/png")  
    


origins = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# At the moment not the safest way to expose app to outside world (we are still in 
# developemnt stage). Maybe use nginx or traefik for this. And use authentication to
# limit access
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("fastapi_rpg:app", host="0.0.0.0", port=8000, reload=False)
