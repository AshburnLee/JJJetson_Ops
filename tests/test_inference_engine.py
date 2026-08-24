"""InferenceEngine skeleton: create/destroy/reset on TransformerModel."""

import inference_engine_me
import transformer_model_me

HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_LAYERS = 2
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 32
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 256


def _create_model(num_layers: int = NUM_LAYERS) -> int:
    return transformer_model_me.create_model(
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        num_layers,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        VOCAB_SIZE,
        MAX_SEQ_LEN,
    )


def test_inference_engine_lifecycle() -> None:
    model = _create_model()
    try:
        engine = inference_engine_me.create_engine(model)
        try:
            assert inference_engine_me.kv_cache_len(engine) == 0
            assert inference_engine_me.next_pos(engine) == 0
            assert inference_engine_me.kv_cache_num_layers(engine) == NUM_LAYERS

            inference_engine_me.reset_engine(engine)
            assert inference_engine_me.kv_cache_len(engine) == 0
            assert inference_engine_me.next_pos(engine) == 0
            print("Passed test_inference_engine_lifecycle")
        finally:
            inference_engine_me.destroy_engine(engine)
    finally:
        transformer_model_me.destroy_model(model)


def test_inference_engine_single_layer_kv() -> None:
    model = _create_model(num_layers=1)
    try:
        engine = inference_engine_me.create_engine(model)
        try:
            assert inference_engine_me.kv_cache_num_layers(engine) == 1
            print("Passed test_inference_engine_single_layer_kv")
        finally:
            inference_engine_me.destroy_engine(engine)
    finally:
        transformer_model_me.destroy_model(model)


if __name__ == "__main__":
    test_inference_engine_lifecycle()
    test_inference_engine_single_layer_kv()
