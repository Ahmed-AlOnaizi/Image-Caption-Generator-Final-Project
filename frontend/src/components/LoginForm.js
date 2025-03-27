// src/LoginForm.js
import React, { useState } from "react";
import { Form, Button } from "react-bootstrap";
import axios from "axios";
import { useNavigate } from "react-router-dom";

function LoginForm() {
  const navigate = useNavigate(); 
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");

  const handleLogin = async () => {
    if (!email || !password) {
      alert("Please enter both email and password.");
      return;
    }
    try {
      const res = await axios.post(
        "http://localhost:5000/api/login",
        { email, password },
        { withCredentials: true } // session cookie
      );
      // If success:
      setMessage(res.data.message || "Logged in");
      
      if (res.data.status === "ok") {
        // redirect to homepage 
        navigate("/");
      }
    } catch (err) {
      console.error(err);
      if (err.response && err.response.data.error) {
        setMessage(err.response.data.error);
      } else {
        setMessage("Login error");
      }
    }
  };

  return (
    <div className="p-4">
      <h2>Login</h2>
      <Form>
        <Form.Group className="mb-3" controlId="loginEmail">
          <Form.Label>Email</Form.Label>
          <Form.Control
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter email"
          />
        </Form.Group>

        <Form.Group className="mb-3" controlId="loginPassword">
          <Form.Label>Password</Form.Label>
          <Form.Control
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter password"
          />
        </Form.Group>

        <Button variant="primary" onClick={handleLogin}>
          Login
        </Button>
      </Form>

      {message && (
        <div className="mt-3 alert alert-info">
          {message}
        </div>
      )}
    </div>
  );
}

export default LoginForm;
