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
  flags.DEFINE_string('output', default = 'ner_dataset', help = 'path to processed dataset directory')

def main(unused_argv):
  tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
  train_samples, eval_samples = list(), list()
  tag_map = {'Disease': 37, 'Gene': 38, 'Protein': 39, 'Enzyme': 40,
             'Var': 41,'MPA': 42, 'Interaction': 43, 'Pathway': 44,
             'CPA': 45, 'Reg': 46, 'PosReg': 47, 'NegReg': 48}
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
        tag = tag_map[d['obj']]
        ner_tags[tokens.index((text[begin:end], (begin, end)))] = tag
      ner_tags = list()
    is_train = np.random.multinomial(1, (9/10,1/10), size = 1)[0,0].astype(np.bool_)
    samples = train_samples if is_train else eval_samples
    samples.append({'tokens': [token[0] for token in tokens], 'ner_tags': ner_tags})
  with open('train.json', 'w') as f:
    f.write(json.dumps({'version': "0.1.0", "data": train_samples}, ensure_ascii = False))
  with open('val.json', 'w') as f:
    f.write(json.dumps({'version': "0.1.0", "data": eval_samples}, ensure_ascii = False))

def load_json():
  dastaset = load_dataset('json',
                          data_files = {
                            'train': 'train.json',
                            'validate': 'val.json',
                            'test': 'val.json'},
                          field = 'data')
  return dataset

if __name__ == "__main__"
  add_options()
  app.run(main)

