// src/components/FeedbackForm.js
import React, { useState } from "react";
import axios from "axios";
import { Form, Button } from "react-bootstrap";

function FeedbackForm() {
  const [feedback, setFeedback] = useState("");

  const handleSubmit = async () => {
    if (!feedback.trim()) {
      alert("Please enter feedback");
      return;
    }
    try {
      await axios.post("http://localhost:5000/api/feedback", {
        feedback: feedback
      });
      alert("Feedback submitted!");
      setFeedback("");
    } catch (err) {
      console.error(err);
      alert("Error submitting feedback");
    }
  };

  return (
    <div className="mt-4">
      <h3>User Feedback</h3>
      <Form>
        <Form.Group className="mb-3">
          <Form.Label>What did you think of the captions?</Form.Label>
          <Form.Control
            as="textarea"
            rows={4}
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
          />
        </Form.Group>
        <Button variant="success" onClick={handleSubmit}>
          Submit Feedback
        </Button>
      </Form>
    </div>
  );
}

export default FeedbackForm;
