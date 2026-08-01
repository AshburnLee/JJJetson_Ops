"""WeightLoader skeleton: ModelConfig validate + load API surface (no format parsing yet)."""

import weight_loader_me


def test_model_config_validate_ok():
    assert weight_loader_me.validate_config(
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=32,
        vocab_size=32000,
        max_seq_len=512,
    )
    print("validate_config ok")


def test_model_config_validate_rejects_bad_heads():
    try:
        weight_loader_me.validate_config(
            hidden_size=128,
            intermediate_size=256,
            num_layers=2,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=31,
            vocab_size=32000,
            max_seq_len=512,
        )
    except ValueError:
        print("validate_config rejects bad head_dim")
        return
    raise AssertionError("expected ValueError for invalid head_dim")


def test_load_fixture_not_implemented():
    try:
        weight_loader_me.load_fixture("/nonexistent/fixture")
    except RuntimeError as exc:
        assert "not implemented" in str(exc)
        print("load_fixture not implemented (expected)")
        return
    raise AssertionError("expected RuntimeError from load_fixture stub")


if __name__ == "__main__":
    test_model_config_validate_ok()
    test_model_config_validate_rejects_bad_heads()
    test_load_fixture_not_implemented()
