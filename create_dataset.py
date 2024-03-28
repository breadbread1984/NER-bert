#!/usr/bin/python3

import json
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, exists, splitext
from absl import flags, app
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

FLAGS = flags.FLAGS

def add_option():
  flags.DEFINE_string('dataset', default = 'dataset', help = 'path to dataset directory')

def main(unused_argv):
  tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
  train_samples, eval_samples = list(), list()
  token_tag_map = {'Disease': 'DISEASE', 'Gene': 'GENE', 'Protein': 'PROTEIN', 'Enzyme': 'ENZYME',
                   'Var': 'VAR','MPA': 'MPA', 'Interaction': 'INTERACTION', 'Pathway': 'PATHWAY',
                   'CPA': 'CPA', 'Reg': 'REG', 'PosReg': 'POSREG', 'NegReg': 'NEGREG'}
  for f in listdir(FLAGS.dataset):
    stem, ext = splitext(f)
    if ext != '.json': continue
    with open(join(FLAGS.dataset, f), 'r') as f:
      sample = json.loads(f.read())
      text = sample['text']
      tokens = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
      ner_tags = ['O' for token in tokens]
      denotations = sample['denotations']
      for d in denotations:
        begin = d['span']['begin']
        end = d['span']['end']
        entity_tokens = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text[begin:end])
        entity_tokens = [(token[0], (token[1][0] + begin, token[1][1] + begin)) for token in entity_tokens]
        for idx, token in enumerate(entity_tokens):
          tag = token_tag_map[d['obj']]
          ner_tags[tokens.index((token[0], (token[1][0], token[1][1])))] = \
            ('S-' + tag) if idx == 0 and idx == len(entity_tokens) - 1 else \
            ('B-' + tag) if idx == 0 else \
            ('E-' + tag) if idx == len(entity_tokens) - 1 else \
            ('I-' + tag)
    is_train = np.random.multinomial(1, (9/10,1/10), size = 1)[0,0].astype(np.bool_)
    samples = train_samples if is_train else eval_samples
    samples.append({'tokens': [token[0] for token in tokens], 'ner_tags': ner_tags})
  with open('train.json', 'w') as f:
    f.write(json.dumps(train_samples, ensure_ascii = False))
  with open('val.json', 'w') as f:
    f.write(json.dumps(eval_samples, ensure_ascii = False))

def load_json():
  dastaset = load_dataset('json',
                          data_files = {
                            'train': 'train.json',
                            'validate': 'val.json',
                            'test': 'val.json'},
                          field = 'data')
  return dataset

if __name__ == "__main__":
  add_option()
  app.run(main)

