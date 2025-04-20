import torch
import torch.nn as nn
from InputEmbeddings import InputEmbeddings
from PositionalEncoding import PositionalEncoding

# Test
token_ids = torch.tensor([1, 2, 3, 4, 5]).unsqueeze(1)
embedding_layer = InputEmbeddings(vocab_size = 10000, d_model = 512)
token_embeddings = embedding_layer(token_ids)
print(token_embeddings[0,0,:10])

pos_encoding_layer = PositionalEncoding(d_model=512, max_seq_length=4)
output = pos_encoding_layer(token_embeddings)
print(output.shape)
print(output[0][0][:10])

# # Test
# # Instantiate InputEmbeddings and apply it to token_ids
# token_ids = torch.tensor([1, 2, 3, 4, 5]).unsqueeze(1)
# embedding_layer = InputEmbeddings(vocab_size = 10000, d_model = 512)
# output = embedding_layer(token_ids)
# print(output.shape)

# # Test FeedForwardSubLayer
# # Instantiate the FeedForwardSubLayer and apply it to x
# feed_forward = FeedForwardSubLayer(d_model, d_ff)
# output = feed_forward(x)
# print(f"Input shape: {x.shape}")
# print(f"Output shape: {output.shape}")