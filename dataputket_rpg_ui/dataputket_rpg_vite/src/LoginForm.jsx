import axios from 'axios';
import { useState } from 'react';
import { Redirect } from 'react-router-dom';

import PropTypes from 'prop-types'
import Form from './Form'
import { Route } from 'react-router-dom/cjs/react-router-dom.min';


function LoginForm() {
    const [username, setUserName] = useState('');   
    const [password, setPassword] = useState('');
    const [loggedIn, setLoggedIn] = useState(false);


    //const history = useHistory();
    

    const handleSubmit = async (event) => {
        //const history = useHistory();
        event.preventDefault();
        
        const response = await fetch('http://34.140.132.125:8000/token', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                'username': username,
                'password': password,	
            }),
        });

        const data = await response.json();
        console.log(data)
        if (response.ok) {
            //setToken(data.access_token);
            console.log(data.access_token)
            sessionStorage.setItem('access_token', data.access_token);  
            setLoggedIn(true);
            //history.push('/dataputket_rpg');
            //return <Form />;
            //return redirect('/dataputket_rpg')

        } else {
            console.error("Invald username or password")

        }
    };

    if (loggedIn) {
        return <Form  /> 
    }
            
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

