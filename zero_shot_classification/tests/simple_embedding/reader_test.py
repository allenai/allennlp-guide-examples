from unittest import TestCase

from zero_shot.simple_embedding.dataset_reader import ZeroShotClassificationReader


class ZeroShotClassificationReaderTest(TestCase):
    def test_read_topic_data(self):
        reader = ZeroShotClassificationReader(label_file="test_fixtures/topic_classes.txt")
        instances = reader.read("test_fixtures/topic_dev.txt")
        assert len(instances) == 7
        assert [t.text for t in instances[0]["text"][:3]] == ["What", "makes", "friendship"]
        assert len(instances[0]["labels"]) == 6  # there are 6 unique labels in the fixture
        labels = [label.tokens for label in instances[0]["labels"]]
        label_strings = [''.join(token.text for token in label) for label in labels]
        expected_label = label_strings.index("Family&Relationships")
        assert expected_label >= 0
        assert instances[0]["gold_label"].sequence_index == expected_label

    def test_read_emotion_data(self):
        reader = ZeroShotClassificationReader()
        instances = reader.read("test_fixtures/emotion_dev.txt")
        assert len(instances) == 7
        assert [t.text for t in instances[0]["text"][:3]] == ["My", "car", "skidded"]
        assert len(instances[0]["labels"]) == 5  # there are 5 unique labels in the fixture
        labels = [label.tokens for label in instances[0]["labels"]]
        label_strings = [''.join(token.text for token in label) for label in labels]
        expected_label = label_strings.index("fear")
        assert expected_label >= 0
        assert instances[0]["gold_label"].sequence_index == expected_label
