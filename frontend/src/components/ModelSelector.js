// src/components/ModelSelector.js
import React from "react";
import { Form } from "react-bootstrap";

function ModelSelector({ modelList, selectedModel, setSelectedModel }) {
  return (
    <Form.Group className="mt-3">
      <Form.Label>Select Model</Form.Label>
      <Form.Select
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
      >
        {modelList.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </Form.Select>
    </Form.Group>
  );
}

export default ModelSelector;
