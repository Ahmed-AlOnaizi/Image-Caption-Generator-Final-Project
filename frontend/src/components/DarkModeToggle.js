import React from "react";
import { Button } from "react-bootstrap";

function DarkModeToggle({ darkMode, setDarkMode }) {
  return (
    <button 
  aria-label="Toggle dark mode" 
  aria-pressed={darkMode}
  onClick={() => setDarkMode(!darkMode)}
>
  {darkMode ? "Light Mode" : "Dark Mode"}
</button>
  );
}

export default DarkModeToggle;