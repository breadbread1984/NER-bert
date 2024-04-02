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

def parse(offset_mapping, token_cls, id2label):
  entities = list()
  status = 'O'
  start = None
  end = None
  for offset, cls in zip(offset_mapping, token_cls):
    label = id2label[cls]
    if status == 'O':
      if label == 'O': continue
      elif label.startswith('B-'):
        status = label
        start = offset[0]
      elif label.startswith('S-'):
        status = label
        start = offset[0]
        end = offset[1]
        entities.append(label[2:], (start, end))
      else:
        print('parse error: %s right after %s!' % (label, status))
        status = 'O'
    elif status.startswith('B-'):
      if label == 'O':
        print('parse error: %s right after %s!' % (label, status))
        status = 'O'
      elif label.startswith('I-'): status = label
      elif label.startswith('E-'):
        status = label
        end = offset[1]
        entities.append(label[2:], (start, end))
      else:
        print('parse error: %s right after %s!' % (label, status))
        status = 'O'
    elif status.startswith('I-'):
      if label == 'O':
        print('parse error: %s right after %s!' % (label, status))
        status = 'O'
      elif label.startswith('I-'): status = label
      elif label.startswith('E-'):
        status = label
        end = offset[1]
        entities.append(label[2:], (start, end))
      else:
        print('parse error: %s right after %s!' % (label, status))
        status = 'O'
    elif status.startswith('E-'):
      if label == 'O': status = label
      elif label.startswith('B-'): status = label
      elif label.startswith('S-'):
        status = label
        start = offset[0]
        end = offset[1]
        entities.append(label[2:], (start, end))
      else:
        print('parse error: %s right after %s!' % (label, status))
        status = 'O'
    elif status.startswith('S-'):
      if label == 'O': status = label
      elif label.startswith('B-'):
        status = label
        start = offset[0]
      elif label.startswith('S-'):
        status = label
        start = offset[0]
        end = offset[1]
        entities.append(label[2:], (start, end))
      else:
        print('parse error: %s right after %s!' % (label, status))
        status = 'O'
    else:
      raise Exception('label: %s, status: %s' % (label, status))
  return entities

def main(unused_argv):
  tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
  model = AutoModelForTokenClassification.from_pretrained(FLAGS.ckpt)
  model.eval()
  model.to(device(FLAGS.device))
  inputs = tokenizer(FLAGS.input, return_tensors = 'pt', padding = True, return_offsets_mapping = True)
  offset_mapping = inputs.pop('offset_mapping') # offset_mapping.shape = (batch, seq_len, 2)
  inputs = inputs.to(device(FLAGS.device))
  outputs = model(**inputs)
  token_cls = torch.argmax(outputs.logits, dim = -1) # token_cls.shape = (batch, seq_len)
  token_cls = token_cls.cpu().numpy()
  print(token_cls)
  print(token_cls.shape)
  with open(join(FLAGS.ckpt, 'config.json'), 'r') as f:
    config = json.loads(f.read())
  id2label = {int(k):v for k,v in config['id2label'].items()}
  entities = parse(offset_mapping[0], token_cls[0], id2label)
  for label, (start, end) in entities:
    print('%s: %s' % (FLAGS.input[start:end], label))

if __name__ == "__main__":
  add_options()
  app.run(main)
