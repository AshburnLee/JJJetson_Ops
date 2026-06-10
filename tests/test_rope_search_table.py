import rope_search_table_me

from rope_test_common import (
    check_all_tokens,
    make_rope_inputs,
)


def test_all():
    input_np, pos_np, output_np, dims = make_rope_inputs()
    rope_search_table_me.RoPE(input_np, pos_np, output_np, dims)
    assert check_all_tokens(input_np, output_np, pos_np, verbose=False), (
        "rope_search_table output differs from NumPy reference"
    )
    print("Passed")


if __name__ == "__main__":
    test_all()
