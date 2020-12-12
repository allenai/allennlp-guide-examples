from collections import defaultdict
from typing import Dict, List

import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register('simple_embedding_zero_shot_classifier')
class EmbeddingZeroShotClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder

        self.accuracy = CategoricalAccuracy()
        self.labelwise_accuracy = defaultdict(lambda: CategoricalAccuracy())

        check_dimensions_match(self.embedder.get_output_dim(), self.encoder.get_output_dim(),
                               "embedder output", "encoder_output")

        # We want to just embed label tokens with the base embedding layer.  The TextFieldEmbedder
        # could actually be BERT or some other contextualizer; this will pull out the bottom,
        # non-contextualized layer from that model.
        self.label_embedder = util.find_embedding_layer(self)

    def forward(self,
                text: Dict[str, torch.Tensor],
                labels: Dict[str, torch.Tensor],
                gold_label: torch.Tensor = None,
                accuracy_key: List[str] = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)

        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)

        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)

        # We just want label ids that we can pass to the label embedder.  This will be a tensor with
        # shape (batch_size, num_labels, num_label_tokens).  Hopefully num_label_tokens is just 1,
        # but with wordpiece tokenizers it could be more than 1.  We'll handle that case with
        # average pooling below.
        label_ids = util.get_token_ids_from_text_field_tensors(labels)

        # Our labels were in a ListField[TextField], which means there's an extra dimension here we
        # need to account for, and tell our utility method about.
        label_mask = util.get_text_field_mask(labels, num_wrapping_dims=1)

        # Shape: (batch_size, num_labels, num_label_tokens, embedding_dim)
        label_token_embeddings = self.label_embedder(label_ids)

        # Shape: (batch_size, num_labels, embedding_dim)
        label_embeddings = util.masked_mean(label_token_embeddings, label_mask.unsqueeze(-1), dim=2)

        # Because we're enforcing that encoding_dim equals embedding_dim, we can just do a dot
        # product here, with torch's batched matrix multiply (bmm) function.
        # Shape: (batch_size, num_labels)
        logits = label_embeddings.bmm(encoded_text.unsqueeze(-1)).squeeze(-1)

        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Shape: (1,)
        output = {'probs': probs}
        if gold_label is not None:
            # IndexField gives a trailing dimension with length 1.
            gold_label = gold_label.squeeze(-1)
            self.accuracy(logits, gold_label)
            if accuracy_key:
                for logit, label, key in zip(logits, gold_label, accuracy_key):
                    self.labelwise_accuracy[key](logit.unsqueeze(0), label.unsqueeze(0))
            output['loss'] = torch.nn.functional.cross_entropy(logits, gold_label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self.accuracy.get_metric(reset)}
        for key in self.labelwise_accuracy:
            metrics[f"_{key}_acc"] = self.labelwise_accuracy[key].get_metric(reset)
        return metrics
