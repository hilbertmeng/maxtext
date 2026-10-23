"""Full-position/chunked observations must implement the quoted definition."""
import unittest
import numpy as np
import jax.numpy as jnp
from head_position_capture import statistics


class PositionStatisticsTest(unittest.TestCase):
    def test_dense_reference_all_positions_and_partial_chunk(self):
        rng = np.random.default_rng(37)
        q = rng.normal(size=(2, 17, 3, 8)).astype(np.float32)
        k = rng.normal(size=q.shape).astype(np.float32)
        tokens = rng.integers(2, 60, size=(2, 17), dtype=np.int32)
        tokens[:, 3] = 0; tokens[:, 6] = 15; tokens[:, 9] = 187
        top, h, ah = map(np.asarray, statistics(jnp.array(q), jnp.array(k), jnp.array(tokens), chunk_size=4))
        logits = np.einsum('bqhd,bshd->bhqs', q.astype(float), k.astype(float))
        valid = np.arange(17)[None, :] <= np.arange(17)[:, None]
        logits = np.where(valid, logits, -1e30)
        np.testing.assert_array_equal(top, logits.argmax(-1))
        def entropy(x, mask):
            x = np.where(mask, x, -1e30)
            e = np.exp(x - x.max(-1, keepdims=True)) * mask
            p = e / np.maximum(e.sum(-1, keepdims=True), 1e-30)
            return -np.sum(p * np.log(np.maximum(p, 1e-30)), -1)
        np.testing.assert_allclose(h, entropy(logits, valid), atol=2e-6)
        allowed = (np.arange(17)[None] >= 2) & ~np.isin(tokens, [0, 15, 187])
        np.testing.assert_allclose(ah, entropy(logits, valid & allowed[:, None, None, :]), atol=2e-6)
        assert h.shape == (2, 3, 17) and np.isfinite(ah).all()

    def test_exact_token_ids_not_all_punctuation_or_query_filter(self):
        # Key 3 dominates from query3 onward. Its token is the only changing item.
        q = jnp.ones((1, 9, 1, 1)); k = jnp.zeros_like(q).at[:, 3].set(100)
        for token_id, expected in [(0,False),(15,False),(187,False),(964,True),(16,True),(535,True)]:
            tokens = jnp.full((1,9),42).at[:,3].set(token_id)
            top, _, _ = statistics(q,k,tokens,chunk_size=4)
            top = np.asarray(top)[0,0]; ids=np.asarray(tokens)[0,top]
            active = (top>=2)&~np.isin(ids,[0,15,187])
            self.assertFalse(active[:2].any())
            self.assertTrue(np.all(active[3:]==expected))


if __name__ == '__main__': unittest.main()
