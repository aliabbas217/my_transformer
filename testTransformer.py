transformer = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
outputs = transformer(input_tokens, src_mask, tgt_mask, cross_mask)
print(outputs)
print(outputs.shape)