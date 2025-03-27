# save_tokenizer.py

import pickle
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Save the tokenizer to disk
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved to 'models/tokenizer.pkl'")
