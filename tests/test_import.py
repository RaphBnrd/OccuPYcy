def test_import_and_api():
    import occupycy

    assert isinstance(occupycy.__version__, str)

    assert hasattr(occupycy, "data")
    assert hasattr(occupycy, "likelihood")
