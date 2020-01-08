local bert_model = "bert-base-uncased";

{
    "dataset_reader" : {
        "type": "classification-tsv",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model,
            }
        }
    },
    "train_data_path": "data/movie_review/train.tsv",
    "validation_data_path": "data/movie_review/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": bert_model,
                    "top_layer_only": true,
                    "requires_grad": false
                }
            }
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": bert_model,
            "requires_grad": false
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 8
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}
