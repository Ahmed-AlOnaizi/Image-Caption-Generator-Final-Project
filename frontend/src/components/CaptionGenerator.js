// src/CaptionGenerator.js
import React, { useState } from "react";
import axios from "axios";
import { Form, Button, Spinner } from "react-bootstrap";
import TextToSpeechButton from "./TextToSpeechButton";

function CaptionGenerator({ selectedModel }) {
  const [imageFile, setImageFile] = useState(null);
  const [captions, setCaptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedCaption, setSelectedCaption] = useState("");

  // States for the final, chosen caption & image preview
  const [savedCaption, setSavedCaption] = useState("");
  const [imagePreviewUrl, setImagePreviewUrl] = useState("");

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImageFile(file);

    
    if (file) {
      const previewUrl = URL.createObjectURL(file);
      setImagePreviewUrl(previewUrl);
    }
  };

  const handleGenerateCaptions = async () => {
    if (!selectedModel || !imageFile) {
      alert("Please select a model and an image first.");
      return;
    }
    try {
      setLoading(true);
      setCaptions([]);
      setSelectedCaption("");
      setSavedCaption(""); 

      const formData = new FormData();
      formData.append("model_name", selectedModel);
      formData.append("image", imageFile);

      const response = await axios.post(
        "http://localhost:5000/api/generate-captions",
        formData
      );
      setCaptions(response.data.captions); 
    } catch (error) {
      console.error(error);
      alert("Error generating captions.");
    } finally {
      setLoading(false);
    }
  };

  // Choose the best caption 
  const handleCaptionSelect = (caption) => {
    setSelectedCaption(caption);
  };

  // Submits the best caption 
  const handleSubmitBestCaption = () => {
    if (!selectedCaption) {
      alert("Please pick a best caption first!");
      return;
    }
    // Save the chosen caption locally
    setSavedCaption(selectedCaption);
    alert(`You selected: "${selectedCaption}" as the best caption.`);

    
    axios
      .post(
        "http://localhost:5000/api/save-caption",
        {
          image_url: imagePreviewUrl, 
          caption: selectedCaption,
        },
        { withCredentials: true }
      )
      .then((res) => {
        console.log("Caption saved successfully:", res.data);
      })
      .catch((err) => {
        console.error("Error saving caption:", err.response?.data?.error || err);
        alert("Error saving caption");
      });
  };

  return (
    <div className="mt-3">
      <h2>Caption Generator</h2>

      <Form>
        {/* File Upload */}
        <Form.Group controlId="uploadFile">
            <Form.Label>Upload Image for Caption</Form.Label>
            <Form.Control
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                aria-describedby="fileHelpBlock"
            />
            <Form.Text id="fileHelpBlock" muted>
            Select an image from your computer to generate captions
            </Form.Text>
        </Form.Group>

        {/* Generate Button */}
        <Button variant="primary" onClick={handleGenerateCaptions}>
          Generate Captions
        </Button>
      </Form>

      {/* Loading Spinner */}
      {loading && (
        <div className="mt-3">
          <Spinner animation="border" role="status" />
          <p>Generating captions, please wait...</p>
        </div>
      )}

      {/* Show the 3 captions as radio options with TTS buttons if not loading */}
      {!loading && captions.length > 0 && (
        <div className="mt-3">
          <h5>Caption Suggestions (pick one):</h5>
          {captions.map((cap, idx) => (
            <div key={idx} className="mb-2">
              <Form.Check
                type="radio"
                label={cap}
                name="captionOptions"
                value={cap}
                checked={selectedCaption === cap}
                onChange={() => handleCaptionSelect(cap)}
                className="d-inline-block me-2"
              />
              <TextToSpeechButton caption={cap} />
            </div>
          ))}

          <Button variant="success" className="mt-2" onClick={handleSubmitBestCaption}>
            Submit Best Caption
          </Button>
        </div>
      )}

      {/* Display the final chosen caption + image preview once user submits */}
      {savedCaption && (
        <div className="mt-4">
          <h4>Final Chosen Caption</h4>
          {imagePreviewUrl && (
            <div>
              <img
                src={imagePreviewUrl}
                alt="Uploaded Preview"
                style={{ maxWidth: "300px", display: "block", marginBottom: "10px" }}
              />
            </div>
          )}
          <p>{savedCaption}</p>
          <TextToSpeechButton caption={savedCaption} />
        </div>
      )}
    </div>
  );
}

export default CaptionGenerator;
