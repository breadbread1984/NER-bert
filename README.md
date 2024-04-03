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
adaseq train -c task.yaml
```

## predict with NER

```shell
python3 predict.py --ckpt ckpt/ner/<id>/output_best --device (cpu|cuda) --input <input string>
```

example:

```shell
python3 predict.py --ckpt ckpt_ner --input "Loss-of-function de novo mutations play an important role in severe human neural tube defects.\nBACKGROUND: Neural tube defects (NTDs) are very common and severe birth defects that are caused by failure of neural tube closure and that have a complex aetiology. Anencephaly and spina bifida are severe NTDs that affect reproductive fitness and suggest a role for de novo mutations (DNMs) in their aetiology.\nMETHODS: We used whole-exome sequencing in 43 sporadic cases affected with myelomeningocele or anencephaly and their unaffected parents to identify DNMs in their exomes.\nRESULTS: We identified 42 coding DNMs in 25 cases, of which 6 were loss of function (LoF) showing a higher rate of LoF DNM in our cohort compared with control cohorts. Notably, we identified two protein-truncating DNMs in two independent cases in SHROOM3, previously associated with NTDs only in animal models. We have demonstrated a significant enrichment of LoF DNMs in this gene in NTDs compared with the gene specific DNM rate and to the DNM rate estimated from control cohorts. We also identified one nonsense DNM in PAX3 and two potentially causative missense DNMs in GRHL3 and PTPRS.\nCONCLUSIONS: Our study demonstrates an important role of LoF DNMs in the development of NTDs and strongly implicates SHROOM3 in its aetiology."
```
