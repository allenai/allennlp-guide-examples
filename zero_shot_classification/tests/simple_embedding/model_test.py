from allennlp.common.testing import ModelTestCase

class EmbeddingZeroShotClassifierTest(ModelTestCase):
    def test_model_can_train_save_and_load(self):
        import zero_shot.simple_embedding  # to get our classes registered
        config_file = "test_fixtures/configs/simple_embedding.json"
        self.set_up_model(config_file, "test_fixtures/topic_dev.txt")
        self.ensure_model_can_train_save_and_load(config_file)
