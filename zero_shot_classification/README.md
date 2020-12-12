Runnable code for the zero-shot classification example in the AllenNLP course.

Code is in the python module `zero_shot`.  Because we like tests, there are also tests in the
`tests/` module, using tiny data files found in `test_fixtures`.  We used these tests _during model
development_, as we were writing the course chapter, to be sure that our implementation actually
worked with a fast test on tiny data, before trying to run anything on full data.  The tests are
intended to be run from the `zero_shot_classification` directory, with pytest.

There is a fair amount of code in here, but most of it is just copied between different versions of
the exercise.  We use several different models in here to do zero-shot classification, and each is a
complete example with code in its own submodule inside `zero_shot`.
