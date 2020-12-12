from typing import Dict, Iterable, List, Set

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import IndexField, ListField, MetadataField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer


@DatasetReader.register("zero_shot_classification")
class ZeroShotClassificationReader(DatasetReader):
    """
    A dataset reader intended for use with the "emotion" and "topic" categories in the data from the
    "Benchmarking Zero-Shot Text Classification" paper, ACL 2019.

    # Parameters

    tokenizer: `Tokenizer`

        Determines how to split strings of text into tokens.

    token_indexers: `Dict[str, TokenIndexer]`

        Determines how strings get represented as tensors before being passed to the model.

    max_tokens: `int`

        If given, we'll truncate input text to this length

    label_file: `str`

        If given, this should be a file containing a list of labels, one label per line.  The topic
        dataset is given with a integers as labels in the actual data, with a separate file
        containing the mapping from those integers to labels.  For convenience, we'll map them to
        their strings here using this file, so our vocabulary works as it's intended.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        label_file: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or SpacyTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.label_mapping: Dict[int, str] = None
        if label_file:
            self.label_mapping = {}
            for i, line in enumerate(open(label_file)):
                self.label_mapping[i] = line.strip()

        # This is for convenience in looking at metrics by label, so we can distinguish dev accuracy
        # on whether we had training data for that label.  We don't know the order of the labels
        # when we get them in the model, and we're just training to predict an index into that list,
        # so we also pass along a string that will let us know what the true label string was.
        self.accuracy_key = {
            "anger": "ang",
            "disgust": "dis",
            "fear": "fea",
            "guilt": "gui",
            "joy": "joy",
            "love": "lov",
            "noemo": "no",
            "sadness": "sad",
            "shame": "sha",
            "surprise": "sur",
            "Society & Culture": "soc",
            "Science & Mathematics": "sci",
            "Health": "hea",
            "Education & Reference": "edu",
            "Computers & Internet": "com",
            "Sports": "spo",
            "Business & Finance": "bus",
            "Entertainment & Music": "ent",
            "Family & Relationships": "fam",
            "Politics & Government": "pol",
        }

    def _read(self, file_path: str) -> Iterable[Instance]:
        # Because we need to get a list of all of the labels that we can pass to the model, we go
        # through the data twice: once to read all the labels, then once to actually create the
        # instances.  We cache what we read the first time.
        instances = []
        label_set: Set[str] = set()
        with open(cached_path(file_path), "r") as lines:
            for line in lines:
                # The emotion dataset has three columns, for some reason, while the topic dataset
                # has two.  This handles both cases.
                parts = line.strip().split("\t")
                label = parts[0]
                text = parts[-1]

                if self.label_mapping:
                    label = self.label_mapping[int(label)]
                label_set.add(label)
                instances.append((label, text))
        all_labels = list(label_set)
        for label, text in instances:
            yield self.text_to_instance(text, all_labels, label)

    def text_to_instance(
        self, text: str, labels: List[str], gold_label: str = None
    ) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[: self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)

        # We'll represent the candidate labels as a list of TextFields, because that's the easiest
        # way to share vocabularies between the text embedding and the labels, which is the whole
        # point of our zero-shot modeling right now.  It'll also be really convenient in later model
        # iterations, as we'll see.
        label_text_fields = []
        for label in labels:
            # Potential optimization: pre-tokenize the labels. But we won't bother here.
            label_tokens = self.tokenizer.tokenize(label)
            label_text_fields.append(TextField(label_tokens, self.token_indexers))
        labels_field = ListField(label_text_fields)
        fields = {"text": text_field, "labels": labels_field}
        if gold_label:
            label_index = labels.index(gold_label)
            if label_index == -1:
                raise ValueError(f"gold label ({gold_label}) not found in given label set!")
            fields["gold_label"] = IndexField(label_index, labels)
            fields["accuracy_key"] = MetadataField(self.accuracy_key[gold_label])
        return Instance(fields)
