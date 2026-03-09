def test_import_and_api():
    import occupycy

    assert isinstance(occupycy.__version__, str)
    assert hasattr(occupycy, "nll_logits")
    assert hasattr(occupycy, "nll_probs")