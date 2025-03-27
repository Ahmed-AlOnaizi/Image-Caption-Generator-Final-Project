// src/App.js

import React, { useState, useEffect } from "react";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

import {
  Navbar,
  Nav,
  Container,
  Row,
  Col
} from "react-bootstrap";

import DarkModeToggle from "./components/DarkModeToggle";
import ModelSelector from "./components/ModelSelector";
import CaptionGenerator from "./components/CaptionGenerator";
import FeedbackForm from "./components/FeedbackForm";

import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import MyCaptionsPage from "./components/MyCaptionsPage"; 
import RegisterForm from "./components/RegisterForm";
import LoginForm from "./components/LoginForm";

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [modelList, setModelList] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");

  
  useEffect(() => {
    axios
      .get("http://localhost:5000/api/models")
      .then((res) => {
        setModelList(res.data);
        if (res.data.length > 0) {
          setSelectedModel(res.data[0]);
        }
      })
      .catch((err) => console.log(err));
  }, []);

  return (
    <Router>
      <div className={darkMode ? "dark-mode full-screen-bg" : "full-screen-bg"}>
        {/* -- NAVIGATION BAR -- */}
        <Navbar
          expand="lg"
          variant={darkMode ? "dark" : "light"}
          bg={darkMode ? "dark" : "light"}
        >
          <Container>
            {/* Logo */}
            <Navbar.Brand as={Link} to="/">
              <img
                src="/logo.png" 
                width="40"
                height="40"
                className="d-inline-block align-top me-2"
                alt="Logo"
              />
              Image Caption Generator
            </Navbar.Brand>

            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
              {/* Left side nav items */}
              <Nav className="me-auto">
                <Nav.Link as={Link} to="/my-captions">
                  My Captions
                </Nav.Link>
              </Nav>

              {/* Right side - dark mode toggle, login/register */}
              <Nav>
                {/* Dark mode button in Nav */}
                <DarkModeToggle darkMode={darkMode} setDarkMode={setDarkMode} />

                {/* Register + Login as links */}
                <Nav.Link as={Link} to="/register" className="ms-2">
                  Register
                </Nav.Link>
                <Nav.Link as={Link} to="/login" className="ms-2">
                  Login
                </Nav.Link>
              </Nav>
            </Navbar.Collapse>
          </Container>
        </Navbar>

        {/* MAIN ROUTES */}
        <Routes>
          {/* HOME PAGE (Model Selection + Caption Generator + Feedback) */}
          <Route
            path="/"
            element={
              <Container className="py-4">
                <Row>
                  <Col md={6} className="mb-4">
                    <ModelSelector
                      modelList={modelList}
                      selectedModel={selectedModel}
                      setSelectedModel={setSelectedModel}
                    />
                    <CaptionGenerator selectedModel={selectedModel} />
                  </Col>
                  <Col md={6}>
                    <FeedbackForm />
                  </Col>
                </Row>
              </Container>
            }
          />

          {/* REGISTER PAGE */}
          <Route path="/register" element={<RegisterForm />} />

          {/* LOGIN PAGE */}
          <Route path="/login" element={<LoginForm />} />

          {/* MY CAPTIONS PAGE */}
          <Route path="/my-captions" element={<MyCaptionsPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
