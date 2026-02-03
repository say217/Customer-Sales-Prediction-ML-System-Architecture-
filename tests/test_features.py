def test_preprocessor_output_shape(preprocessor, X_sample):
    Xt = preprocessor.fit_transform(X_sample)
    assert Xt.shape[0] == X_sample.shape[0]
