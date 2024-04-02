#!/usr/bin/python3

import json
from os.path import join, exists
from absl import flags, app
import torch
from torch import device
from transformers import AutoTokenizer, AutoModelForTokenClassification

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'input string')
  flags.DEFINE_string('ckpt', default = 'ckpt_ner', help = 'path to checkpoint directory')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def main(unused_argv):
  tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
  model = AutoModelForTokenClassification.from_pretrained(FLAGS.ckpt)
  model.eval()
  model.to(device(FLAGS.device))
  inputs = tokenizer(FLAGS.input, return_tensors = 'pt', padding = True, return_offsets_mapping = True)
  offset_mapping = inputs.pop('offset_mapping')
  inputs = inputs.to(device(FLAGS.device))
  outputs = model(**inputs)
  token_cls = torch.argmax(outputs.logits, dim = -1)
  print(token_cls)
  print(token_cls.shape)
  with open(join(FLAGS.ckpt, 'config.json'), 'r') as f:
    config = json.loads(f.read())
  id2label = {int(k):v for k,v in config['id2label'].items()}
  label2id = config['label2id']

if __name__ == "__main__":
  add_options()
  app.run(main)
