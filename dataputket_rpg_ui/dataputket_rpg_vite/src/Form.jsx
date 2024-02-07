import React, { useState } from'react';
import axios from 'axios';

const FormComponent = () => {
    const [user_prompt, setUserPrompt] = useState('');
    const [version_number, setVersionNumber] = useState('');
    const [steps, setSteps] = useState('');
    const [model_name, setModelName] = useState('');
    const [activate, setActivate] = useState('');
    const [use_base, setUseBase] = useState('');
    const [base_ratio, setBaseRatio] = useState('');
    const [base_prompt, setBasePrompt] = useState('');
    const [batch_size, setBatchSize] = useState('');
    const [seed, setSeed] = useState('');
    const [cfg, setCfg] = useState('');
    const [width, setWidth] = useState('');
    const [height, setHeight] = useState('');
    
    const handleSubmit = async (event) => {
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

        try {
            const response = await axios({
                method: 'post',
                url: 'http://localhost:8000/dataputket_rpg',
                data: payload,
                headers: { 'Content-Type': 'application/json'}
            });
            console.log(response.data);
        } catch (error) {
            console.log(error);
        }
    };

    return (
        <form onSubmit={handleSubmit} className="space-y-4">
            <div>
                <label htmlFor="user_promp" className="block text-sm font-medium test-gray-700">User Prompt</label>
                <input id='user_promp' name='user_prompt' type="text" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="version_number" className="block text-sm font-medium test-gray-700">Version Number</label>
                <input id='version_number' name='version_number' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="steps" className="block text-sm font-medium test-gray-700">Steps</label>
                <input id='steps' name='steps' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="model_name" className="block text-sm font-medium test-gray-700">Model Name</label>
                <input id='model_name' name='model_name' type="text" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="activate" className="block text-sm font-medium test-gray-700">Activate</label>
                <input id='activate' name='activate' type="checkbox" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="use_base" className="block text-sm font-medium test-gray-700">Use Base</label>
                <input id='use_base' name='use_base' type="checkbox" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="base_ratio" className="block text-sm font-medium test-gray-700">Base Ratio</label>
                <input id='base_ratio' name='base_ratio' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="base_prompt" className="block text-sm font-medium test-gray-700">Base Prompt</label>
                <input id='base_prompt' name='base_prompt' type="text" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="batch_size" className="block text-sm font-medium test-gray-700">Batch Size</label>
                <input id='batch_size' name='batch_size' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="seed" className="block text-sm font-medium test-gray-700">Seed</label>
                <input id='seed' name='seed' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="cfg" className="block text-sm font-medium test-gray-700">Cfg</label>
                <input id='cfg' name='cfg' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="width" className="block text-sm font-medium test-gray-700">Image Width</label>
                <input id='width' name='width' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="height" className="block text-sm font-medium test-gray-700">Image Height</label>
                <input id='height' name='height' type="number" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={name} onChange={(e) => setName(e.target.value)} />
            </div>

            <button type="submit" className="w-full py-2 px-4 border border-transparent rounded-md text-white bg-indigo-600 hover:bg-indigo-700">Submit</button>
        </form>
    );
};

export default FormComponent;
