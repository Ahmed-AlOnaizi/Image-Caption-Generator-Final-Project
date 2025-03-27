// src/components/TextToSpeechButton.js
import React from "react";
import { Button } from "react-bootstrap";

function TextToSpeechButton({ caption }) {
  const speakCaption = () => {
    const utter = new SpeechSynthesisUtterance(caption);
    utter.lang = "en-US";
    window.speechSynthesis.speak(utter);
  };

  return (
    <Button variant="outline-secondary" onClick={speakCaption}>
      🔊 Speak Caption
    </Button>
  );
}

export default TextToSpeechButton;
