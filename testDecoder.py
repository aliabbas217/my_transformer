# Instantiate a decoder transformer and apply it to input_tokens and tgt_mask
transformer_decoder = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length)   
output = transformer_decoder(input_tokens, tgt_mask)
print(output)
print(output.shape)