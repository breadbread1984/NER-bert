#!/usr/bin/python3

import torch
from torch import nn
from torchcrf import CRF


class BERT_CRF(nn.Module):
    def __init__(self, model):
        super(BERT_CRF, self).__init__()
        self.encoder = model
        self.config = self.encoder.config
        self.crf = CRF(num_tags = self.config.num_labels, batch_first = True)
    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, position_ids = None, head_mask = None, inputs_embeds = None, labels = None, output_attentions = None, output_hidden_states = None):
        outputs = self.encoder(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, position_ids = position_ids, head_mask = head_mask, inputs_embeds = inputs_embeds, output_attentions = output_attentions, output_hidden_states = output_hidden_states)
        logits = outputs.logits
        loss = None
        if labels is not None:
            # NOTE: input token sequence as a leading token [CLS] and an ending token [SEP], these two special tokens should be removed
            str_lens = torch.sum(attention_mask, dim = -1) # str_lens.shape = (batch)
            mask = attention_mask.to(torch.bool)
            for i in range(str_lens.shape[0]):
                mask[i,0] = False # [CLS]
                mask[i,str_lens[i] - 1] = False # [SEP]
            tags = torch.where(mask, labels, torch.zeros_like(labels))

            log_likelihood, tags = self.crf(emissions = logits[:,1:], tags = tags[:,1:], mask = mask[:,1:]), self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
        tags = torch.Tensor(tags)
        output = (tags,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
