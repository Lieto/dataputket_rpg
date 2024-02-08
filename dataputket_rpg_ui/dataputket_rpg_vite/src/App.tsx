import React, { useState } from'react'
import './App.css'
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import LoginForm from './LoginForm';
import Form from './Form';


function App() {
  const [token, setToken] = useState();

  if (!token) {
    return <LoginForm setToken={setToken} />
  }

  return (
    <div className="wrapper">
      <h1>Application</h1>
      <BrowserRouter>
        <Switch>
          <Route path="/dataputket_rpg">
            <Form />
          </Route>
        </Switch>
      </BrowserRouter>
    </div>
  );
}

export default App
