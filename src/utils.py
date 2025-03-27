
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load BERT
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

def get_bert_score(caption):
    """
    Get BERT fluency score for a caption.
    """
    inputs = bert_tokenizer(caption, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return -outputs.loss.item()

def rank_and_select_best_caption(captions):
    """
    Select the best caption using BERT scores.
    """
    scores = [(caption, get_bert_score(caption)) for caption in captions]
    return sorted(scores, key=lambda x: x[1], reverse=True)[0][0]
