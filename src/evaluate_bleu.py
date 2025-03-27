
from nltk.translate.bleu_score import corpus_bleu
from src.preprocessing import load_captions

# Load test captions 
reference_captions = load_captions('data/dataset/flickr30k_images/results.csv')
generated_captions = [] 

# BLEU score calculation
references = []
hypotheses = []

for img_id, ref_captions in reference_captions.items():
    if img_id in generated_captions:
        references.append([ref.split() for ref in ref_captions])  
        hypotheses.append(generated_captions[img_id].split())     

# Calculate BLEU scores
bleu_1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

print(f"BLEU-1: {bleu_1}")
print(f"BLEU-2: {bleu_2}")
print(f"BLEU-3: {bleu_3}")
print(f"BLEU-4: {bleu_4}")
