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

import argparse
import sys

import t5
import t5x
import gin
import jax.numpy as jnp
import numpy as np

from pyserini.encode import DocumentEncoder, QueryEncoder
from pyserini.encode import JsonlRepresentationWriter, FaissRepresentationWriter, JsonlCollectionIterator
from transformers import T5Tokenizer



training_config_gin_file = "/home/mac/t5x_retrieval/t5x_retrieval/configs/models/de_t5_base.gin"
dtype='bfloat16'
restore_mode='specific'

def _load_model(checkpoint_path: str):
    # Parse config file
    gin.parse_config_file(training_config_gin_file)
    gin.finalize()

    # Get model
    model_config_ref = gin.query_parameter("%MODEL")
    model = model_config_ref.scoped_configurable_fn()

    # Load checkpoint
    t5x_checkpoint = t5x.checkpoints.load_t5x_checkpoint(checkpoint_path)
    model._inference_mode = 'encode'
    return t5x_checkpoint, model



class SentenceT5DocumentEncoder(DocumentEncoder):
    def __init__(self, model_name="gs://t5-data/pretrained_models/t5x/retrieval/gtr_base", tokenizer_name="/home/mac/sentencepiece.model", device='cuda:0'):
        self.device = device
        self.checkpoint, self.model = _load_model(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name or model_name)

    def encode(self, texts, titles=None,  max_length=512, **kwargs):
        inputs = self.tokenizer(
            texts,
            max_length=max_length,
            padding='longest',
            truncation=True,
            add_special_tokens=True
        )
        inputs = jnp.array(inputs["input_ids"])
        input_batch = {
            'left_encoder_input_tokens': inputs
        }

        output = self.model.score_batch(self.checkpoint['target'], input_batch)
        return np.array(output)



encoder_class_map = {
    "st5": SentenceT5DocumentEncoder
}

def init_encoder(encoder, encoder_class, device):
    _encoder_class = encoder_class

    # determine encoder_class
    encoder_class = encoder_class_map[encoder_class]

    # prepare arguments to encoder class
    kwargs = dict(model_name=encoder, device=device)
    if (_encoder_class == "sentence-transformers") or ("sentence-transformers" in encoder):
        kwargs.update(dict(pooling='mean', l2_norm=True))

    return encoder_class(**kwargs)


def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='sub-commands')
    input_parser = commands.add_parser('input')
    input_parser.add_argument('--corpus', type=str,
                              help='directory that contains corpus files to be encoded, in jsonl format.',
                              required=True)
    input_parser.add_argument('--fields', help='fields that contents in jsonl has (in order)',
                              nargs='+', default=['text'], required=False)
    input_parser.add_argument('--delimiter', help='delimiter for the fields', default='\n', required=False)
    input_parser.add_argument('--shard-id', type=int, help='shard-id 0-based', default=0, required=False)
    input_parser.add_argument('--shard-num', type=int, help='number of shards', default=1, required=False)

    output_parser = commands.add_parser('output')
    output_parser.add_argument('--embeddings', type=str, help='directory to store encoded corpus', required=True)
    output_parser.add_argument('--to-faiss', action='store_true', default=False)

    encoder_parser = commands.add_parser('encoder')
    encoder_parser.add_argument('--encoder', type=str, help='encoder name or path', required=False)
    encoder_parser.add_argument('--encoder-class', type=str, required=False, default="st5",
                                choices=["dpr", "bpr", "tct_colbert", "ance", "sentence-transformers", "auto", "st5"],
                                help='which query encoder class to use. `default` would infer from the args.encoder')
    encoder_parser.add_argument('--fields', help='fields to encode', nargs='+', default=['text'], required=False)
    encoder_parser.add_argument('--batch-size', type=int, help='batch size', default=64, required=False)
    encoder_parser.add_argument('--max-length', type=int, help='max length', default=512, required=False)
    encoder_parser.add_argument('--dimension', type=int, help='dimension', default=768, required=False)
    encoder_parser.add_argument('--add-sep', action='store_true', default=False)

    args = parse_args(parser, commands)
    delimiter = args.input.delimiter.replace("\\n", "\n")  # argparse would add \ prior to the passed '\n\n'

    encoder = init_encoder(args.encoder.encoder, args.encoder.encoder_class, device=args.encoder.device)
    if args.output.to_faiss:
        embedding_writer = FaissRepresentationWriter(args.output.embeddings, dimension=args.encoder.dimension)
    else:
        embedding_writer = JsonlRepresentationWriter(args.output.embeddings)
    collection_iterator = JsonlCollectionIterator(args.input.corpus, args.input.fields, delimiter)

    with embedding_writer:
        for batch_info in collection_iterator(args.encoder.batch_size, args.input.shard_id, args.input.shard_num):
            kwargs = {
                'texts': batch_info['text'],
                'titles': batch_info['title'] if 'title' in args.encoder.fields else None,
                'expands': batch_info['expand'] if 'expand' in args.encoder.fields else None,
                'max_length': args.encoder.max_length,
                'add_sep': args.encoder.add_sep,
            }
            embeddings = encoder.encode(**kwargs)
            batch_info['vector'] = embeddings
            embedding_writer.write(batch_info, args.input.fields)
