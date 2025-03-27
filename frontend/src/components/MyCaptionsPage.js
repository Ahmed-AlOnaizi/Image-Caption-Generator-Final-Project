// src/MyCaptionsPage.js
import React, { useEffect, useState } from "react";
import axios from "axios";
import { Container, Row, Col, Card } from "react-bootstrap";

function MyCaptionsPage() {
  const [captions, setCaptions] = useState([]);
  const [error, setError] = useState("");

  
  // Note: withCredentials is required to send the session cookie.
  useEffect(() => {
    axios
      .get("http://localhost:5000/api/my-captions", { withCredentials: true })
      .then((res) => {
        if (res.data.captions) {
          setCaptions(res.data.captions);
        } else if (res.data.error) {
          setError(res.data.error);
        }
      })
      .catch((err) => {
        console.error(err);
        if (err.response && err.response.data.error) {
          setError(err.response.data.error);
        } else {
          setError("Error fetching captions");
        }
      });
  }, []);

  return (
    <Container className="py-4">
      <h2>My Captions</h2>
      {error && <p className="text-danger">{error}</p>}
      {captions.length === 0 && !error && (
        <p>You have no saved captions yet.</p>
      )}
      <Row>
        {captions.map((cap) => (
          <Col md={4} key={cap.id} className="mb-4">
            <Card>
              {cap.image_url && (
                <Card.Img variant="top" src={cap.image_url} alt="User Upload" />
              )}
              <Card.Body>
                <Card.Text>{cap.caption}</Card.Text>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>
    </Container>
  );
}

export default MyCaptionsPage;
