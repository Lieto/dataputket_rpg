import React, { useState } from'react'
import './App.css'
import { BrowserRouter, Router, Route, Switch } from 'react-router-dom';
import LoginForm from './LoginForm';
import Form from './Form';

function useToken() {
  const getToken = () => {
    const tokenString = localStorage.getItem('token');
    const userToken = JSON.parse(tokenString);
    return userToken?.token
  };

  const [token, setToken] = useState(getToken());

  const saveToken = userToken => {
    localStorage.setItem('token', JSON.stringify(userToken));
    setToken(userToken.token);
  };

  return {
    setToken: saveToken,
    token
  }

}
function App() {

  //const { token, setToken } = useToken(); 
  const token = sessionStorage.getItem('access-token');
  if (!token) {
    return <LoginForm />
  }

  return (
    <div className="wrapper">
      <h1>Application</h1>
      <Router>
        <Switch>
        <Route path="/dataputket_rpg" component={Form} />
        </Switch>
      </Router>
    </div>
  );
}

export default App
