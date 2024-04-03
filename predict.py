#!/usr/bin/python3

import json
from os import getcwd
from os.path import join, exists, abspath
from absl import flags, app
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'input string')
  flags.DEFINE_string('ckpt', default = 'ckpt_ner', help = 'path to checkpoint directory')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def main(unused_argv):
  p = pipeline(Tasks.named_entity_recognition, abspath(FLAGS.ckpt))
  results = p(FLAGS.input)
  print(results)

if __name__ == "__main__":
  add_options()
  app.run(main)
