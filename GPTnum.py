import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  
    return g * x + b  
def linear(x, w, b): 
    return x @ w + b
def ffn(x, c_fc, c_proj):  
    a = gelu(linear(x, **c_fc)) 
    x = linear(a, **c_proj)  
    return x
def attention(q, k, v, mask):  
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v
def mha(x, c_attn, c_proj, n_head):  
    x = linear(x, **c_attn)  
    qkv = np.split(x, 3, axis=-1) 
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)] 
    x = np.hstack(out_heads) 
    x = linear(x, **c_proj) 
    return x
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  
    x = x + ffn(layer_norm(x, **ln_2), **mlp) 
    return x
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head): 
    x = wte[inputs] + wpe[range(len(inputs))] 
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  
    x = layer_norm(x, **ln_f) 
    return x @ wte.T  
def generate(inputs, params, n_head, n_tokens_to_generate):
    for _ in range(n_tokens_to_generate):  
        logits = gpt2(inputs, **params, n_head=n_head) 
        next_id = np.argmax(logits[-1]) 
        inputs.append(int(next_id))  
    return inputs[len(inputs) - n_tokens_to_generate :] 
from functools import lru_cache
@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs
import json
import os
import regex as re

class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    def bpe(self, token):
        if token in self.cache: return self.cache[token]
        word, get = tuple(token), get_pairs
        while True:
            bigram = min(get(word), key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks: break
            first, second, new_word, i = bigram[0], bigram[1], [], 0
            while i < len(word):
                try: j = word.index(first, i); new_word.extend(word[i:j]); i = j
                except: new_word.extend(word[i:]); break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second: new_word.append(first + second); i += 2
                else: new_word.append(word[i]); i += 1
            word = tuple(new_word)
            if len(word) == 1: break
        self.cache[token] = word = ' '.join(word)
        return word
    def encode(self, text):
        return [self.encoder[bpe_token] for token in re.findall(self.pat, text) for bpe_token in self.bpe(''.join(self.byte_encoder[b] for b in token.encode('utf-8'))).split(' ')]

    def decode(self, tokens):
        return bytearray([self.byte_decoder[c] for c in ''.join([self.decoder[token] for token in tokens])]).decode('utf-8', errors=self.errors)
def get_encoder():
    with open("./model/encoder.json", 'r') as f:
        encoder = json.load(f)
    with open("./model/vocab.bpe", 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )
def load_gpt2_params(hparams):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d
    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    ckpt = []
    with open("./model/names.json") as f:
        names = json.load(f)
    for name in names:
        path_shape = "./" + name + "shape" + ".json"
        with open(path_shape, "r") as f:
            _ = json.load(f)
        ckpt.append((name, _))
    for name, _ in ckpt:
        path_var = "./" + name + "var" + ".npy"
        array = np.load(path_var)
        name = name[len("model/") :]
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)
    return params
def load_encoder_hparams_and_params():
    encoder = get_encoder()
    hparams = json.load(open("./model/hparams.json"))
    params = load_gpt2_params_from(hparams)
    return encoder, hparams, params
def main():
    encoder, hparams, params = load_encoder_hparams_and_params()
    prompt = input("Prompt: ")
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + 20 < hparams["n_ctx"]
    output_ids = generate(input_ids, params, hparams["n_head"], 20)
    output_text = encoder.decode(output_ids)
    print(output_text)
if __name__ == "__main__":
    main()

