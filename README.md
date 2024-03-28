# Introduction

this project is to do named entity recogintion.

# Usage

## install prerequisite packages

```shell
python3 -m pip install -r requirementst.txt
```

## generate dataset

```shell
python3 create_dataset.py --dataset dataset
```

Upon generating dataset successfully, two json files are generated under current directory.

## train BERT to do NER

```shell
python3 run_ner.py \
  --model_name_or_path google-bert/bert-base-cased \
  --config_name config.json \
  --train_file train.json \
  --validation_file val.json \
  --output_dir ./ckpt_ner \
  --do_train \
  --do_eval
```

**NOTE**: run_ner.py is directly adopted from [transformers](https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py)
