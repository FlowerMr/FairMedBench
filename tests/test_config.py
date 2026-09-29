from fairmedbench.models.factory import build_model


def test_model_factory():
    model = build_model("tiny_cnn", 3, image_size=64)
    assert sum(p.numel() for p in model.parameters()) > 0
