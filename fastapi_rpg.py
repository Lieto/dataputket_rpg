# The code is importing necessary modules and libraries for the application. Here's a
# breakdown of what each import statement does:
import os
import sys
import time

import torch
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm    
from fastapi_login import LoginManager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse 
from loguru import logger
from pydantic import BaseModel
from typing import Annotated



# Use dotenv to store secret /private keys to apis (OpenAI GPT-4). At the moment you 
# have to copy '.env' file to server to use it
# `load_dotenv()` is a function from the `dotenv` library in Python. It is used to load
# the environment variables from a `.env` file into the current environment.
load_dotenv()



SECRET = os.getenv("SECRET")
manager = LoginManager(SECRET, tokenUrl="/auth/token")
fake_users_db = {
    "dataputket_test": {
        "username": "dataputket_test",
        "hashed_password": "fakehashedsecret",
    }
}

def fake_hash_password(password: str):
    return "fakehashed" + password

@manager.user_loader
def load_user(username: str):
    user_dict = fake_users_db.get(username)
    if user_dict:
        return user_dict    
    


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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    
# Add router to app
router = APIRouter(prefix="/dataputket_rpg")

@router.post("/auth/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    hashed_password = user_dict["hashed_password"]
    if not manager.verify_password(form_data.password, hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = manager.create_access_token(
        data={"sub": form_data.username}
    )
    return {"access_token": access_token, "token_type": "bearer"}


# Post request to generate image from prompt
@router.post("/")
async def rpg(token: Annotated[str, Depends(oauth2_scheme)], params: AppParams):
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
    

# Finally create application, add router and run app
app = FastAPI()
app.include_router(router)

origins = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allowed_origins=origins,
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
