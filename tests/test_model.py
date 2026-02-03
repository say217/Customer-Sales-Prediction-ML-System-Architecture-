def test_model_predicts(model, X_sample):
    preds = model.predict(X_sample)
    assert len(preds) == len(X_sample)
