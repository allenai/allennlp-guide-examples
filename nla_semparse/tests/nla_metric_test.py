from nla_semparse.nla_metric import NlaMetric


def test_metric_basic():
    metric = NlaMetric()
    metric([['2']], [['2']])
    assert metric.get_metric() == {"well_formedness": 1.0,
                                   "denotation_accuracy": 1.0,
                                   "sequence_accuracy": 1.0}
    metric.reset()


def test_metric_one_operation():
    metric = NlaMetric()
    metric([['(', '+', '2', '3', ')']], [['(', '+', '2', '3', ')']])
    assert metric.get_metric() == {"well_formedness": 1.0,
                                   "denotation_accuracy": 1.0,
                                   "sequence_accuracy": 1.0}
    metric.reset()
    metric([['(', '+', '2', '3', ')']], [['5']])
    assert metric.get_metric() == {"well_formedness": 1.0,
                                   "denotation_accuracy": 1.0,
                                   "sequence_accuracy": 0.0}
    metric.reset()
    metric([['(', '+', '2', '3', ')']], [['(', '+', '1', '4', ')']])
    assert metric.get_metric() == {"well_formedness": 1.0,
                                   "denotation_accuracy": 1.0,
                                   "sequence_accuracy": 0.0}
    metric.reset()
    metric([['(', '+', '2', '3', ')']], [['(', '-', '1', '4', ')']])
    assert metric.get_metric() == {"well_formedness": 1.0,
                                   "denotation_accuracy": 0.0,
                                   "sequence_accuracy": 0.0}
    metric.reset()


def test_metric_ill_formed_sequences():
    metric = NlaMetric()
    metric([['(', '+', '2', ')']], [['(', '+', '2', '3', ')']])
    assert metric.get_metric() == {"well_formedness": 0.0,
                                   "denotation_accuracy": 0.0,
                                   "sequence_accuracy": 0.0}
    metric.reset()
    metric([['(', '+', ')', ')']], [['(', '+', '2', '3', ')']])
    assert metric.get_metric() == {"well_formedness": 0.0,
                                   "denotation_accuracy": 0.0,
                                   "sequence_accuracy": 0.0}
    metric.reset()
    metric([['(', ')']], [['(', '+', '2', '3', ')']])
    assert metric.get_metric() == {"well_formedness": 0.0,
                                   "denotation_accuracy": 0.0,
                                   "sequence_accuracy": 0.0}
    metric.reset()


def test_metric_real_cases():
    predictions1 = [['(', '-', '(', '*', '(', '(', '(', '(', '(',
                     '(', '(', '(', '(', '(', ')', ')', ')', ')', ')', ')'],
                    ['(', '-', '(', '+', '(', '(', '*', '(', '(',
                     '(', '(', ')', ')', ')', ')', ')', ')', ')', ')', ')']]
    predictions2 = [['132'], ['9']]
    predictions3 = [['(', '-', '(', '*', '(', '(', '(', '(', '(',
                     '(', '(', '(', '(', '(', ')', ')', ')', ')', ')', ')'],
                    ['9']]
    targets = [['(', '+', '(', '+', '(', '*', '5', '2', ')', '(',
                '/', '2', '7', ')', ')', '(', '+', '(', '+', '7',
                '7', ')', '(', '*', '3', '(', '*', '6', '6', ')', ')', ')', ')'],
               ['(', '-', '(', '+', '8', '7', ')', '(', '-', '(',
                '+', '(', '+', '6', '(', '/', '7', '7', ')', ')', '7',
                ')', '(', '*', '(', '/', '5', '4', ')', '8', ')', ')', ')']]
    metric = NlaMetric()
    metric(predictions1, targets)
    assert metric.get_metric() == {"well_formedness": 0.0,
                                   "denotation_accuracy": 0.0,
                                   "sequence_accuracy": 0.0}
    metric.reset()
    metric(predictions2, targets)
    assert metric.get_metric() == {"well_formedness": 1.0,
                                   "denotation_accuracy": 1.0,
                                   "sequence_accuracy": 0.0}
    metric.reset()
    metric(predictions3, targets)
    assert metric.get_metric() == {"well_formedness": 0.5,
                                   "denotation_accuracy": 0.5,
                                   "sequence_accuracy": 0.0}
    metric.reset()
    metric(targets, targets)
    assert metric.get_metric() == {"well_formedness": 1.0,
                                   "denotation_accuracy": 1.0,
                                   "sequence_accuracy": 1.0}
    metric.reset()
