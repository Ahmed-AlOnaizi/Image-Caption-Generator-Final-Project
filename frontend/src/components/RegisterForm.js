// src/RegisterForm.js
import React, { useState } from 'react';
import axios from 'axios';
import { Form, Button } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';

function RegisterForm() {
  const navigate = useNavigate(); // <-- from react-router-dom

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleRegister = async () => {
    try {
      const res = await axios.post("http://localhost:5000/api/register", {
        email,
        password
      });
      alert(res.data.message);

      // If register is successful, redirect to main page 
      if (res.data.status === "ok") {
        navigate("/"); 
      }
    } catch (err) {
      console.error(err);
      alert(err.response?.data?.error || "Registration error");
    }
  };

  return (
    <div className="p-4">
      <h2>Register</h2>
      <Form>
        <Form.Group className="mb-3">
          <Form.Label>Email</Form.Label>
          <Form.Control 
            type="email" 
            value={email} 
            onChange={e => setEmail(e.target.value)} />
        </Form.Group>

        <Form.Group className="mb-3">
          <Form.Label>Password</Form.Label>
          <Form.Control 
            type="password" 
            value={password} 
            onChange={e => setPassword(e.target.value)} />
        </Form.Group>

        <Button variant="primary" onClick={handleRegister}>
          Register
        </Button>
      </Form>
    </div>
  );
}

export default RegisterForm;
