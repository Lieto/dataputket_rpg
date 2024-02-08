import os 
import sys
import time
import torch 
from mllm import GPT4
from modules import errors, extensions
from loguru import logger 

# The code is adding the paths of three repositories (`stablediffusion`, `k-diffusion`,
# and `generative-models`) to the system path. This allows the code to import modules and
# files from these repositories. By adding these paths to the system path, the code
# ensures that the Python interpreter can find and access the necessary files and modules
# from these repositories.

sys.path.append('/app/repositories/stablediffusion')
sys.path.append('/app/repositories/k-diffusion')
sys.path.append('/app/repositories/generative-models')

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
    
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
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

