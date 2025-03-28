# Image Caption Generator Web App

A web application that automatically generates textual descriptions for images using multiple deep learning backends (ResNet, InceptionV3) and a BERT tokenizer. The tool focuses on **accessibility** (text-to-speech, keyboard-only navigation) and provides a user-friendly interface for generating, saving, and reporting captions.
Note: A more comprehensive Readme can be found in the appendix of the dissertation. This one is very simple and is not that detailed
## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Technologies](#technologies)    
4. [Setup & Installation](#setup--installation)  
5. [Usage Instructions](#usage-instructions)  
6. [Model Training & Data](#model-training--data)  
7. [Accessibility Notes](#accessibility-notes)  
  


---

## Overview
This project addresses the **problem** of providing accurate, real-time image captions in a user-friendly environment. By integrating different CNN backbones (e.g., **ResNet** for speed, **InceptionV3** for balanced accuracy), the tool allows users to upload images and receive immediate textual descriptions. It also includes a **text-to-speech (TTS)** feature for visually impaired users and a **reporting** mechanism to gather feedback on poor captions.

### Core Objectives:
- Real-time captioning (<5s ideally)  
- Simple, accessible UI with TTS and keyboard navigation  
- Multiple model backends for user experimentation  
- Basic user account system (register, login)  
- Feedback loops (report inaccurate captions)

---

## Features
- **Multiple Models**: ResNet, InceptionV3 available for image feature extraction.  
- **Text-to-Speech**: Reads aloud the generated caption.  
- **User Registration & Login**: Each user can save chosen captions to a personal “My Captions” page.  
- **Report Button**: Submits reasons like “very inaccurate,” “repetitive words,” or “offensive,” plus an “other” field for custom feedback.  
- **GPU & CPU**: Can run on CPU but is faster on GPU, especially for training and heavy inference.

---

## Technologies
- **Python** & **Flask** for the backend  
- **TensorFlow/Keras** for CNN and text decoding  
- **BERT Tokenizer** (Hugging Face) for subword tokenization  
- **React** (JavaScript) for the front end  
- **SQLite** (or another DB) for user accounts and saved captions  
- **Bootstrap/React-Bootstrap** (optional) for styling

---
## Setup & Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/your-username/image-caption-generator.git

2. Set up environment
cd image-caption-generator/
pip install -r requirements.txt (OR pip install -r requirements-dev.txt)


Initialize Database (if using SQLite):

python
-- from app import db
-- db.create_all()
-- exit()

Frontend Installation (if using React):
cd ../frontend
npm install

---
## Usage Instructions

1. Run the backend
cd image-caption-generator/web_app
python app.py

2. Run the frontend
Run the frontend (React):
cd ../frontend
npm start

The frontend typically appears at http://localhost:3000.

Register & Login:

In the browser, go to http://localhost:3000.

Create an account, then log in.

---
## Model Selection & Upload:

Pick a model (e.g., InceptionV3).

Upload an image.

Wait a few seconds for the caption to generate.

Submit Best Caption:

Choose your favorite among the suggestions.

Click Submit Best Caption to save it under “My Captions.”

Report Issues:

If a caption is very inaccurate or offensive, click Report.

Fill out the short form (check boxes, optional “Other”).

Submit the report.

Check “My Captions”:

See a gallery of previously accepted captions.

(Optional) remove them using a Delete button if implemented.

---
## Model Training & Data
Dataset: Flickr30k (30k images). Some partial tests with MSCOCO ~40k.

Training:

CPU: ~18–24 hours for ~14 epochs

GPU: ~2–4 hours for the same data (even when adding more epochs)


to train run the train.py file
