# The code is importing necessary modules and libraries for the application. Here's a
# breakdown of what each import statement does:
import os
import sys
import time

import torch
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse 
from loguru import logger
from pydantic import BaseModel

from mllm import GPT4
from modules import errors, extensions

# The code is adding the paths of three repositories (`stablediffusion`, `k-diffusion`,
# and `generative-models`) to the system path. This allows the code to import modules and
# files from these repositories. By adding these paths to the system path, the code
# ensures that the Python interpreter can find and access the necessary files and modules
# from these repositories.

sys.path.append('/app/repositories/stablediffusion')
sys.path.append('/app/repositories/k-diffusion')
sys.path.append('/app/repositories/generative-models')

# Use dotenv to store secret /private keys to apis (OpenAI GPT-4). At the moment you 
# have to copy '.env' file to server to use it
# `load_dotenv()` is a function from the `dotenv` library in Python. It is used to load
# the environment variables from a `.env` file into the current environment.
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def initialize(model_name=None):
    """
    The `initialize` function initializes various modules and loads models for a stable
    diffusion model in Python.
    
    :param model_name: The `model_name` parameter is used to specify the name of the model
    that you want to load. It is an optional parameter, so if you don't provide a value, the
    function will not load any specific model
    """
    from modules import options, shared, shared_options
    from modules.shared import cmd_opts
    shared.options_templates = shared_options.options_templates #{}
    shared.opts = options.Options(shared_options.options_templates, shared_options.restricted_opts)
    shared.restricted_opts = shared_options.restricted_opts
    if os.path.exists(shared.config_filename):
        shared.opts.load(shared.config_filename)
    extensions.list_extensions()
    
    
    from modules import devices
    devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = \
        (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])

    devices.dtype = torch.float32 if cmd_opts.no_half else torch.float16
    devices.dtype_vae = torch.float32 if cmd_opts.no_half or cmd_opts.no_half_vae else torch.float16

    shared.device = devices.device
    shared.weight_load_location = None if cmd_opts.lowram else "cpu"
    from modules import shared_state
    shared.state = shared_state.State()

    from modules import styles
    shared.prompt_styles = styles.StyleDatabase(shared.styles_filename)

    from modules import interrogate
    shared.interrogator = interrogate.InterrogateModels("interrogate")

    from modules import shared_total_tqdm
    shared.total_tqdm = shared_total_tqdm.TotalTQDM()

    from modules import devices, memmon
    shared.mem_mon = memmon.MemUsageMonitor("MemMon", devices.device, shared.opts)
    shared.mem_mon.start()


    import modules.sd_models
    modules.sd_models.setup_model() # load models
    modules.sd_models.list_models()
    #startup_timer.record("list SD models")
 
    modules.scripts.load_scripts()
    modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
    
    #startup_timer.record("load scripts")
    print('txt2img_scripts',modules.scripts.scripts_txt2img.scripts)
 
    try:
        modules.sd_models.load_model(model_name=model_name)
    except Exception as e:
        errors.display(e, "loading stable diffusion model")
        print("", file=sys.stderr)
        print("Stable diffusion model failed to load, exiting", file=sys.stderr)
        exit(1)


def RPG(user_prompt, 
        diffusion_model, 
        version, 
        split_ratio, 
        activate=True, 
        use_base=False, 
        base_ratio=0, 
        base_prompt=None,
        batch_size=1,
        seed=1234,
        use_personalized=False,
        cfg=5,
        steps=20,
        height=1024,
        width=1024):
    
    # Get layout and regional description prompt for distincs areas from openai api gpt-4
    openai_api_key = OPENAI_API_KEY  

    import modules.txt2img
    input_prompt = user_prompt
    params = GPT4(input_prompt,version,openai_api_key)
    
    regional_prompt = params['Regional Prompt']
    split_ratio = params['split ratio']
    
    logger.debug(f"regional prompt: {regional_prompt}")
    logger.debug(F"split ratio: {split_ratio}")
    
    if use_base:
        if base_prompt is None:
            regional_prompt = user_prompt + '\n' + regional_prompt  
        else:
            regional_prompt = base_prompt + '\n' + regional_prompt  
    
    # Regional settings:
    regional_settings = { 
                         'activate':activate, 
                         'split_ratio':split_ratio,
                         'base_ratio': base_ratio,
                         'use_base':use_base,
                         'use_common':False}
    
    logger.debug(f"regional settings: {regional_settings}") 
    
    # Use txt2img to get image from prompt. Notice: this can take a while at the moment 
    # depending on the prompt length, resolution and the number of iterations. 
    # (Tesla K80 GPU: 20 iters, 1024x1024, 50-100 characters user prompt -> 4 mins)
    image, _, _, _ = modules.txt2img.txt2img(
        id_task="task", 
        prompt=regional_prompt,
        negative_prompt="",
        prompt_styles=[],
        steps=steps, 
        sampler_index=0,    
        restore_faces=False,
        tiling=False,
        n_iter=1,
        batch_size=batch_size,
        cfg_scale=cfg,  
        seed=seed, # -1 means random, choose a number larger than 0 to get a deterministic result   
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0, 
        seed_resize_from_w=0,
        seed_enable_extras=False,
        height=height, 
        width=width, 
        enable_hr=False, 
        denoising_strength=0.7, 
        hr_scale=0, 
        hr_upscaler="Latent",
        hr_second_pass_steps=0, 
        hr_resize_x=0, 
        hr_resize_y=0, 
        override_settings_texts=[], 
        **regional_settings,
        )
    
    return image 


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

# Add router to app
router = APIRouter(prefix="/dataputket_rpg")


# Post request to generate image from prompt
@router.post("/")
def rpg(params: AppParams):
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
