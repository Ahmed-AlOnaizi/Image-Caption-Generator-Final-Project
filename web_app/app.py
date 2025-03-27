# backend/app.py
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from transformers import BertTokenizer
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


app = Flask(__name__)
CORS(app, supports_credentials= True)  # Allow cross-origin requests from React

app.config['SECRET_KEY'] = 'SOME_SECRET_KEY'
# DB config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydb.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

script_dir = os.path.dirname(os.path.realpath(__file__))
model_rel_path1 = os.path.join(script_dir, "models", "image_captioning_model.h5")
model_rel_path2 = os.path.join(script_dir, "models", "image_captioning_model.h5")
model_rel_path3 = os.path.join(script_dir, "models", "image_captioning_model.h5")


# Load  models
MODEL_PATHS = {
    "InceptionV3 (Mostly Accurate...)": "../models/image_captioning_model.h5",
    "InceptionV3 (Sometimes Accurate...)": "../models/image_captioning_model_inceptionv3.h5",
    "ResNet (Very quick generating...)": "../models/resnet2_new.h5"
}


# Load tokenizer
tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "models", "tokenizer.pkl")
import pickle
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', 
    unk_token='[UNK]',
    
)


inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def extract_features_inception(img_path):
    img = load_img(img_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = inception_model.predict(img_array)
    return features  


def generate_caption_greedy(model, image_features, max_len=30):
    in_text = ["[CLS]"]
    for _ in range(max_len):
        sequence = tokenizer.encode(in_text, add_special_tokens=False)
        padded_seq = pad_sequences([sequence], maxlen=max_len)
        preds = model.predict([image_features, padded_seq], verbose=0)[0]
        next_id = np.argmax(preds)
        next_word = tokenizer.decode([next_id])
        if next_word == "[SEP]":
            break
        in_text.append(next_word)
    return " ".join(in_text[1:]).replace("[SEP]", "").strip()


def generate_caption_beam(model, image_features, max_len=30, beam_width=3):
    start_token = "[CLS]"
    end_token = "[SEP]"
    sequences = [(start_token, 0.0)]

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            seq_ids = tokenizer.encode(seq, add_special_tokens=False)
            padded_seq = pad_sequences([seq_ids], maxlen=max_len)
            preds = model.predict([image_features, padded_seq], verbose=0)[0]
            
            top_indices = np.argsort(preds)[-beam_width:]
            for idx in top_indices:
                word = tokenizer.decode([idx])
                new_seq = seq + " " + word
                new_score = score + np.log(preds[idx])  
                all_candidates.append((new_seq, new_score))
        # sort by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]

        
        if any(end_token in s[0].split() for s in sequences):
            break

    best_seq, best_score = sequences[0]
    
    return best_seq.replace("[CLS]", "").replace("[SEP]", "").strip()


def generate_caption_topk(model, image_features, max_len=30, k=5, temperature=1.0):
    in_text = ["[CLS]"]
    for _ in range(max_len):
        seq_ids = tokenizer.encode(in_text, add_special_tokens=False)
        padded_seq = pad_sequences([seq_ids], maxlen=max_len)
        preds = model.predict([image_features, padded_seq], verbose=0)[0]
        
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        top_k_indices = np.argsort(preds)[-k:]
        top_k_probs = preds[top_k_indices]
        top_k_probs /= np.sum(top_k_probs)
        next_id = np.random.choice(top_k_indices, p=top_k_probs)
        next_word = tokenizer.decode([next_id])
        if next_word == "[SEP]":
            break
        in_text.append(next_word)
    return " ".join(in_text[1:]).replace("[SEP]", "").strip()



@app.route('/api/models', methods=['GET'])
def list_models():
    """Return list of model names."""
    return jsonify(list(MODEL_PATHS.keys()))

@app.route('/api/generate-captions', methods=['POST'])
def generate_captions():
    """
    Expects:
      - model_name (str)
      - image (file, 'multipart/form-data')
    Returns:
      - JSON with {"captions": [caption1, caption2, caption3]}
    """
    model_name = request.form.get('model_name')
    if model_name not in MODEL_PATHS:
        return jsonify({"error": "Invalid model name"}), 400

    model_path = MODEL_PATHS[model_name]
    # Load the selected model
    caption_model = load_model(model_path)

    # Save uploaded file temporarily
    file = request.files['image']
    temp_path = "temp.jpg"
    file.save(temp_path)

    # Extract features 
    image_features = extract_features_inception(temp_path)

    # Generate three different captions
    caption1 = generate_caption_greedy(caption_model, image_features)
    caption2 = generate_caption_beam(caption_model, image_features)
    caption3 = generate_caption_topk(caption_model, image_features)

    # Cleanup
    os.remove(temp_path)

    return jsonify({
        "captions": [caption1, caption2, caption3]
    })

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    user_feedback = data.get('feedback', '')
    #  store user_feedback in a DB or log it
    print("User Feedback:", user_feedback)
    return jsonify({"status": "ok"})

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

class CaptionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_url = db.Column(db.String(255), nullable=True)
    caption = db.Column(db.String(500), nullable=False)

    user = db.relationship('User', backref=db.backref('captions', lazy=True))


@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Check if user already exists
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"error": "User already exists"}), 400

    # Hash the password
    pw_hash = generate_password_hash(password)

    new_user = User(email=email, password_hash=pw_hash)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"status": "ok", "message": "User registered successfully"})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401

    # Check password hash
    if not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid credentials"}), 401

    # Save user ID in session
    session['user_id'] = user.id
    return jsonify({"status": "ok", "message": "Logged in"})


@app.route('/api/save-caption', methods=['POST'])
def save_caption():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401

    user_id = session['user_id']
    data = request.json
    image_url = data.get('image_url')
    caption = data.get('caption')

    
    print(f"Attempting to save caption for user {user_id}: image_url={image_url}, caption={caption}")

    new_record = CaptionRecord(user_id=user_id, image_url=image_url, caption=caption)
    db.session.add(new_record)
    db.session.commit()

    print("Caption saved to database.")

    return jsonify({"status": "ok", "message": "Caption saved"})

@app.route('/api/my-captions', methods=['GET'])
def my_captions():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401

    user_id = session['user_id']
    records = CaptionRecord.query.filter_by(user_id=user_id).all()

    
    result = []
    for r in records:
        result.append({
            "id": r.id,
            "image_url": r.image_url,
            "caption": r.caption
        })
    return jsonify({"captions": result})




if __name__ == '__main__':
    
    
    app.run(debug=True, port=5000)