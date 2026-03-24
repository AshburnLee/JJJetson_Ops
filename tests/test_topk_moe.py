import torch
import numpy as np
import top_k_moe_me

def test_top_k_moe():
    torch.manual_seed(24)
    dtype = torch.float32

    cases = [
        (4,    32,  3),
        (128,  8,   2),
        (4096, 512, 2),
        (512,  64,  3),
        (1024, 128, 4),
    ]

    for n_tokens, n_experts, topk in cases:
        print(f"\nTesting: tokens={n_tokens}, experts={n_experts}, topk={topk}")

        logits = torch.randn(n_tokens, n_experts, device="cpu", dtype=dtype) * 10
        assert logits.is_contiguous()

        # torch as reference
        topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
        
        weights_ref = torch.softmax(topk_logits, dim=-1)
        ids_ref = topk_indices.to(torch.int32)

        # my output
        logits_me = logits.cpu().numpy()
        weights_me = np.zeros((n_tokens, topk), dtype=np.float32)
        ids_me = np.zeros((n_tokens, topk), dtype=np.int32)
        
        top_k_moe_me.top_k_moe(
            logits=logits_me,
            topk=topk,
            weights=weights_me,
            ids=ids_me,
            input_dims=[n_tokens, n_experts]
        )
        
        # compare
        weights_me = torch.from_numpy(weights_me)
        ids_me = torch.from_numpy(ids_me)
        atol = 1e-4 if dtype == torch.float32 else 1e-3
        rtol = 1e-5

        weights_ok = torch.allclose(weights_me, weights_ref, atol=atol, rtol=rtol)
        ids_ok     = torch.equal(ids_me, ids_ref)

        print(f"  weights match: {weights_ok}")
        print(f"  ids     match: {ids_ok}")

        if not weights_ok:
            diff = (weights_me - weights_ref).abs()
            print(f"  max weight diff: {diff.max().item():.2e}")
        assert weights_ok and ids_ok, "Test failed!"

    print("\nAll tests passed")


def debug():
    logits_np = np.random.randn(4096, 32).astype(np.float32)
    weights_np = np.zeros((4096, 2), dtype=np.float32)
    ids_np     = np.zeros((4096, 2), dtype=np.int32)

    top_k_moe_me.top_k_moe(
        logits=logits_np,
        topk=2,
        weights=weights_np,
        ids=ids_np,
        input_dims=[4096,32]
    )

    print(weights_np.shape)
    print(ids_np.shape)


if __name__ == "__main__":
    # debug()
    test_top_k_moe()
    
