import t5
import t5x
import gin
import jax.numpy as jnp
import numpy as np

from pyserini.encode import DocumentEncoder, QueryEncoder
from transformers import T5Tokenizer



training_config_gin_file = "config.gin"
checkpoint_path='/Users/mac/Documents/odunayo/t5x_retrieval/20220929/checkpoint_1005400'
dtype='bfloat16'
restore_mode='specific'

def _load_model():
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
    def __init__(self, model_name=None, tokenizer_name=None, device='cuda:0'):
        self.device = device
        self.checkpoint, self.model = _load_model()
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name or model_name)

    def encode(self, texts, titles=None,  max_length=256, **kwargs):
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


#
# if __name__ == "__main__":
#     encoder = SentenceT5DocumentEncoder(model_name="/Users/mac/Documents/odunayo/t5x_retrieval/20220929/checkpoint_1005400",
#                                         tokenizer_name="/Users/mac/Documents/odunayo/sentencepiece.model")
#     source_sentence = ["That is a happy person", "That is a happy person"]
#     out = encoder.encode(source_sentence)
#
#     print(type(out))