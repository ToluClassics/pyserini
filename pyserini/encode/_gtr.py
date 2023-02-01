#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer
from typing import Optional, Tuple, Union

from pyserini.encode import DocumentEncoder, QueryEncoder

class GTRModel(nn.Module):
    def __init__(self, input_dim=768, output_dim=768):
        super(GTRModel, self).__init__()
        self.t5_components = T5EncoderModel.from_pretrained("t5-base")
        self.linear_q = nn.Linear(768, 768, bias=False)

    def forward(self, input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None):
        outputs = self.t5_components(input_ids, attention_mask)
        embeddings = self._mean_pooling(outputs, attention_mask)
        embeddings = self.linear_q(embeddings)
        return F.normalize(embeddings, p=2, dim=1)
    
    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class GtrDocumentEncoder(DocumentEncoder):
    def __init__(self, model_name, tokenizer_name=None, device='cuda:0', pooling='mean'):
        self.device = device
        gtrModel = GTRModel()
        self.model = torch.load(model_name)
        self.model.to(self.device)

        try:
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        except:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name or model_name, use_fast=False)

        self.has_model = True
        self.pooling = pooling

    def encode(self, texts, titles=None, max_length=512, add_sep=False, **kwargs):
        shared_tokenizer_kwargs = dict(
            max_length=max_length,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        input_kwargs = {}
        if not add_sep:
            input_kwargs["text"] = [f'{title} {text}' for title, text in zip(titles, texts)] if titles is not None else texts
        else:
            if titles is not None:
                input_kwargs["text"] = titles
                input_kwargs["text_pair"] = texts
            else:
                input_kwargs["text"] = texts

        inputs = self.tokenizer(**input_kwargs, **shared_tokenizer_kwargs)
        inputs.to(self.device)
        embeddings = self.model(**inputs)

        return embeddings.cpu().detach().numpy()

class GtrQueryEncoder(QueryEncoder):
    def __init__(self, model_name, tokenizer_name=None, device='cuda:0', pooling='mean'):
        self.device = device
        gtrModel = GTRModel()
        self.model = torch.load(model_name)
        self.model.to(self.device)

        try:
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        except:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name or model_name, use_fast=False)

        self.has_model = True
        self.pooling = pooling

    def encode(self, texts, titles=None, max_length=64, add_sep=False, **kwargs):
        shared_tokenizer_kwargs = dict(
            max_length=max_length,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        input_kwargs = {}
        if not add_sep:
            input_kwargs["text"] = [f'{title} {text}' for title, text in zip(titles, texts)] if titles is not None else texts
        else:
            if titles is not None:
                input_kwargs["text"] = titles
                input_kwargs["text_pair"] = texts
            else:
                input_kwargs["text"] = texts

        inputs = self.tokenizer(**input_kwargs, **shared_tokenizer_kwargs)
        inputs.to(self.device)
        embeddings = self.model(**inputs)

        return embeddings

if __name__ == '__main__':
    encoder = GtrDocumentEncoder('/home/oogundep/odunayo/gtr-base-pth')
    doc = 'That is a very happy person'
    encoded_doc = encoder.encode([doc])
    print(encoded_doc.shape)
    print(encoded_doc)