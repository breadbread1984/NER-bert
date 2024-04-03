#!/usr/bin/python3

import json
from os import getcwd
from os.path import join, exists, abspath
from absl import flags, app
import adaseq
import huggingface
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'input string')
  flags.DEFINE_string('ckpt', default = 'ckpt_ner', help = 'path to checkpoint directory')
  flags.DEFINE_enum('device', default = 'gpu', enum_values = {'gpu', 'cpu'}, help = 'device to use')

def main(unused_argv):
  print('load checkpoint: %s\n' % abspath(FLAGS.ckpt))
  p = pipeline(Tasks.named_entity_recognition, abspath(FLAGS.ckpt), device = FLAGS.device)
  results = p(FLAGS.input)
  print(results)

if __name__ == "__main__":
  add_options()
  app.run(main)
