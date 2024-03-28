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
  single_token_tag_map = {'Disease': 37, 'Gene': 38, 'Protein': 39, 'Enzyme': 40,
                          'Var': 41,'MPA': 42, 'Interaction': 43, 'Pathway': 44,
                          'CPA': 45, 'Reg': 46, 'PosReg': 47, 'NegReg': 48}
  beg_token_tag_map = {'Disease': 1, 'Gene': 2, 'Protein': 3, 'Enzyme': 4,
                       'Var': 5,'MPA': 6, 'Interaction': 7, 'Pathway': 8,
                       'CPA': 9, 'Reg': 10, 'PosReg': 11, 'NegReg': 12}
  int_token_tag_map = {'Disease': 13, 'Gene': 14, 'Protein': 15, 'Enzyme': 16,
                       'Var': 17,'MPA': 18, 'Interaction': 19, 'Pathway': 20,
                       'CPA': 21, 'Reg': 22, 'PosReg': 23, 'NegReg': 24}
  end_token_tag_map = {'Disease': 25, 'Gene': 26, 'Protein': 27, 'Enzyme': 28,
                       'Var': 29,'MPA': 30, 'Interaction': 31, 'Pathway': 32,
                       'CPA': 33, 'Reg': 34, 'PosReg': 35, 'NegReg': 36}
  for f in listdir(FLAGS.dataset):
    stem, ext = splitext(f)
    if ext != '.json': continue
    with open(join(FLAGS.dataset, f), 'r') as f:
      sample = json.loads(f.read())
      text = sample['text']
      tokens = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
      ner_tags = [0 for token in tokens]
      denotations = sample['denotations']
      for d in denotations:
        begin = d['span']['begin']
        end = d['span']['end']
        entity_tokens = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text[begin:end])
        entity_tokens = [(token[0], (token[1][0] + begin, token[1][1] + begin)) for token in entity_tokens]
        for idx, token in enumerate(entity_tokens):
          ner_tags[tokens.index((token[0], (token[1][0], token[1][1])))] = \
            single_token_tag_map[d['obj']] if idx == 0 and idx == len(entity_tokens) - 1 else \
            beg_token_tag_map[d['obj']] if idx == 0 else \
            end_token_tag_map[d['obj']] if idx == len(entity_tokens) - 1 else \
            int_token_tag_map[d['obj']]
    is_train = np.random.multinomial(1, (9/10,1/10), size = 1)[0,0].astype(np.bool_)
    samples = train_samples if is_train else eval_samples
    samples.append({'tokens': [token[0] for token in tokens], 'ner_tags': ner_tags})
  with open('train.json', 'w') as f:
    f.write(json.dumps({'version': "0.1.0", "features": train_samples}, ensure_ascii = False))
  with open('val.json', 'w') as f:
    f.write(json.dumps({'version': "0.1.0", "features": eval_samples}, ensure_ascii = False))

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

