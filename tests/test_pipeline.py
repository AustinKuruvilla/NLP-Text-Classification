def test_preprocess_text():
    from src.pipeline import preprocess_text
    assert preprocess_text("Hello, WORLD!!") == "hello world"


def test_build_demo_dataset():
    from src.pipeline import build_demo_dataset
    df = build_demo_dataset()
    assert "text" in df.columns and "label" in df.columns
