import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import convert2prompt

def main():
    example_file_path = '../data/translation/example_rules.txt'
    prompt = convert2prompt(example_file_path)
    with open('../data/translation/prompt.txt', 'w', encoding="utf-8") as f:
        f.write(prompt)

if __name__ == '__main__':
    main()