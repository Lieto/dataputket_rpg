import axios from 'axios';
import { useState } from'react';
import { useHistory } from'react-router-dom';

function LoginForm() {
    const [username, setUserName] = useState('');   
    const [password, setPassword] = useState('');
    const history = useHistory();

    const handleSubmit = async (event) => {
        event.preventDefault();

        try {
            const response = await axios.post('http://localhost:8000/auth/token', {
                username: username,
                password: password ,     
            });
           

            console.log(response.data.access_token);
            localStorage('access_token', response.data.access_token);
            history.pushState('/dataputket_rpg', null, '/dataputket_rpg');
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <form onSubmit={handleSubmit} className='space-y-4'>
            <div>
                <label htmlFor="username" className="block text-sm font-medium test-gray-700">Username</label>
                <input id="username" name="username" type="text" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={username} onChange={(e) => setUserName(e.target.value)} />
            </div> 
            <div>
                <label htmlFor="password" className="block text-sm font-medium test-gray-700">Password</label>
                <input type='password' value={password} onChange={(e) => setPassword(e.target.value)} />
                <input id="password" name="password" type="password" required className='mt-1 block w-full p-2 border border-gray-300 rounded-md' value={password} onChange={(e) => setPassword(e.target.value)} /> 
            </div>
            <button type="submit" className="w-full py-2 px-4 border border-transparent rounded-md text-white bg-indigo-600 hover:bg-indigo-700">Login</button>
        </form>
    );
}

export default LoginForm;
