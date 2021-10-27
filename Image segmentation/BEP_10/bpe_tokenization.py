import re, collections

def get_vocab(file):
    vocab = collections.defaultdict(int)
    with open(file, 'r', encoding='utf-8') as fhand:
        for line in fhand:
            words = line.strip().split()
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens




def main():
    text= get_vocab("short_text.txt")
    tokens = get_tokens(text)
    print(tokens)
    print('Number of tokens: {}'.format(len(tokens)))
    print("=====")
    
    num_merges = 1000
    for i in range(num_merges):
        pairs = get_stats(text)
        if not pairs:
            print('Iter: {}'.format(i))
            #print('Best pair: {}'.format(best))
            print(tokens)
            print('Number of tokens: {}'.format(len(tokens)))
            print("=====")
            break

        best = max(pairs, key=pairs.get)  
        text = merge_vocab(best, text)
        tokens = get_tokens(text)
            
        
if __name__=="__main__":
    main()
    
#https://leimao.github.io/blog/Byte-Pair-Encoding/
#https://github.com/huggingface/tokenizers