import React, { useState, useEffect } from'react';
import axios from 'axios';

const ImageCanvas = () => {
    const [imageData, setImageData ] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
};

    
const FormComponent = () => {
    const [user_prompt, setUserPrompt] = useState('Red Ferrari on Italian city street');
    const [version_number, setVersionNumber] = useState(0);
    const [steps, setSteps] = useState(10);
    const [model_name, setModelName] = useState("albedobaseXL_v20.safetensors");
    const [activate, setActivate] = useState(1);
    const [use_base, setUseBase] = useState(0);
    const [base_ratio, setBaseRatio] = useState(0.3);
    const [base_prompt, setBasePrompt] = useState('');
    const [batch_size, setBatchSize] = useState(1);
    const [seed, setSeed] = useState(1234);
    const [cfg, setCfg] = useState(5);
    const [width, setWidth] = useState(1024);
    const [height, setHeight] = useState(1024);
    
    const [imageData, setImageData ] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
   
    const handleSubmit = async (event) => {
        setLoading(true);
        setError(null);

        event.preventDefault();

        const payload = {
            user_prompt: user_prompt,
            version_number: version_number,
            steps: steps,
            model_name: model_name,
            activate: activate,
            use_base: use_base,
            base_ratio: base_ratio,
            base_prompt: base_prompt,
            batch_size: batch_size,
            seed: seed,
            cfg: cfg,
            width: width,
            height: height
        };

        const token = sessionStorage.getItem('access_token');
        console.log(token);
        try {
            const response = await axios({
                method: 'post',
                url: 'http://34.140.132.125:8000/dataputket_rpg',
                data: payload,
                headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json'}, 
                responseType: 'blob',


            });
            console.log(response.data);
            const reader = new FileReader();
            reader.onload = (event) => setImageData(event.target.result);
            reader.readAsDataURL(response.data);

        } catch (error) {
            console.log(error);
            setError('Error fetchning result: ' + error.message);
        } finally {
            setLoading(false);  
        }
    };


    return (
        <form onSubmit={handleSubmit} className="space-y-4">
            <div>
                <label htmlFor="user_prompt" className="block text-sm font-medium test-gray-700">User Prompt</label>
                <input id='user_prompt' name='user_prompt' type="text" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={user_prompt} onChange={(e) => setUserPrompt(e.target.value)} />
            </div>
            <div>
                <label htmlFor="version_number" className="block text-sm font-medium test-gray-700">Version Number</label>
                <input id='version_number' name='version_number' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={version_number} onChange={(e) => setVersionNumber(e.target.value)} />
            </div>
            <div>
                <label htmlFor="steps" className="block text-sm font-medium test-gray-700">Steps</label>
                <input id='steps' name='steps' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={steps} onChange={(e) => setSteps(e.target.value)} />
            </div>
            <div>
                <label htmlFor="model_name" className="block text-sm font-medium test-gray-700">Model Name</label>
                <input id='model_name' name='model_name' type="text" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={model_name} onChange={(e) => setModelName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="activate" className="block text-sm font-medium test-gray-700">Activate</label>
                <input id='activate' name='activate' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={activate} onChange={(e) => setActivate(e.target.value)} />
            </div>
            <div>
                <label htmlFor="use_base" className="block text-sm font-medium test-gray-700">Use Base</label>
                <input id='use_base' name='use_base' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={use_base} onChange={(e) => setUseBase(e.target.value)} />
            </div>
            <div>
                <label htmlFor="base_ratio" className="block text-sm font-medium test-gray-700">Base Ratio</label>
                <input id='base_ratio' name='base_ratio' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={base_ratio} onChange={(e) => setBaseRatio(e.target.value)} />
            </div>
            <div>
                <label htmlFor="base_prompt" className="block text-sm font-medium test-gray-700">Base Prompt</label>
                <input id='base_prompt' name='base_prompt' type="text" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={base_prompt} onChange={(e) => setBasePrompt(e.target.value)} />
            </div>
            <div>
                <label htmlFor="batch_size" className="block text-sm font-medium test-gray-700">Batch Size</label>
                <input id='batch_size' name='batch_size' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={batch_size} onChange={(e) => setBatchSize(e.target.value)} />
            </div>
            <div>
                <label htmlFor="seed" className="block text-sm font-medium test-gray-700">Seed</label>
                <input id='seed' name='seed' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={seed} onChange={(e) => setSeed(e.target.value)} />
            </div>
            <div>
                <label htmlFor="cfg" className="block text-sm font-medium test-gray-700">Cfg</label>
                <input id='cfg' name='cfg' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={cfg} onChange={(e) => setCfg(e.target.value)} />
            </div>
            <div>
                <label htmlFor="width" className="block text-sm font-medium test-gray-700">Image Width</label>
                <input id='width' name='width' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={width} onChange={(e) => setWidth(e.target.value)} />
            </div>
            <div>
                <label htmlFor="height" className="block text-sm font-medium test-gray-700">Image Height</label>
                <input id='height' name='height' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={height} onChange={(e) => setHeight(e.target.value)} />
            </div>

            <button type="submit" className="w-full py-2 px-4 border border-transparent rounded-md text-white bg-indigo-600 hover:bg-indigo-700">Submit</button>

            <div>
                { loading ? (
                    <p>Loading image...</p>
                ) : error ? (
                    <p>{ error }</p>
                ) : (
                <img src={imageData} alt="Image from FastAPI" />
                
                ) }
            </div>
        
        </form>
    )
                };

export default FormComponent;

