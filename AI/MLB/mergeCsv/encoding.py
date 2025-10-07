import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os
import re



final_df = pd.read_csv('final_merged_data.csv')  

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def bert_encode_text(text):
    if pd.isna(text) or str(text).strip() == '' or str(text).strip().upper() == 'N/A':
        return np.zeros(model.config.hidden_size)
    with torch.no_grad():
        inputs = tokenizer(
            str(text),
            return_tensors='pt',
            truncation=True,
            max_length=32,
            padding='max_length'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding

# List of columns to encode (must be present in final_df)
umpire_cols = ['umpire_HP'] if 'umpire_HP' in final_df.columns else []
win_loss_cols = [col for col in ['home_Win', 'home_Loss', 'away_Win', 'away_Loss'] if col in final_df.columns]
text_cols_to_encode = umpire_cols + win_loss_cols

print(f"Encoding {len(text_cols_to_encode)} text columns.")

for col in text_cols_to_encode:
    print(f"Encoding column: {col}")
    emb_matrix = np.vstack(final_df[col].apply(bert_encode_text).tolist())  # (num_rows, 768)
    print(f"Applying PCA for {col} (to 1D)...")
    pca = PCA(n_components=1)
    reduced_emb = pca.fit_transform(emb_matrix)  # (num_rows, 1)
    # Replace the original column (keep same name, same position)
    final_df[col] = reduced_emb.flatten()

print("Replaced original text columns with 1-dimensional BERT embeddings, names unchanged.")
final_df.to_csv('final_merged_data.csv', index=False)
print("Success!!!!!!")