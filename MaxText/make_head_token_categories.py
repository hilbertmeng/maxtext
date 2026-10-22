"""Build exclusive, tokenizer-specific lexical classes for attention diagnostics."""
import argparse
import json
from pathlib import Path
import unicodedata
import numpy as np
from tokenizers import Tokenizer


def classify(text, special=False):
    if special:
        return 3
    text = text.strip()
    if not text:
        return 2
    if all(unicodedata.category(c).startswith('P') for c in text):
        return 1
    if all(unicodedata.category(c)[0] in 'PS' for c in text):
        return 4
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tokenizer_json')
    parser.add_argument('output')
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    specials = {t['id'] for t in json.loads(tokenizer.to_str())['added_tokens'] if t.get('special')}
    table = np.array([classify(tokenizer.decode([i], skip_special_tokens=False), i in specials)
                      for i in range(tokenizer.get_vocab_size())], dtype=np.int32)
    np.save(Path(args.output), table)
    print('category_counts', np.bincount(table).tolist())


if __name__ == '__main__':
    main()
