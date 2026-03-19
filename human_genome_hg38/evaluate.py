import numpy as np

def decode(tokens):
    vocab_inv = {0:'A', 1:'T', 2:'G', 3:'C', 4:'N'}
    return ''.join(vocab_inv[t] for t in tokens)

def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

generated = np.load("samples/generated.npy")
for seq_tokens in generated[:10]:
    seq = decode(seq_tokens)
    print(f"GC={gc_content(seq):.2%}  {seq[:60]}...")
