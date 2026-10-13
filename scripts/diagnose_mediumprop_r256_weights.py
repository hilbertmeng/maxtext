"""Read-only partial checkpoint restore for the three BAM R256 down projections.

Run with JAX_PLATFORMS=cpu. Saved TPU sharding is deliberately overridden.
Output arrays retain scan axes; analysis must use the global restored shapes,
not Orbax's per-device write_shape metadata.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import jax
import numpy as np
import orbax.checkpoint as ocp
from flax.traverse_util import flatten_dict, unflatten_dict

PORTS = {"full_m_read_down", "P_loc_down", "mlp_address_down"}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    args = ap.parse_args()
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(args.checkpoint)
    tree = getattr(metadata, "item_metadata", metadata)
    tree = getattr(tree, "tree", tree)
    flat = flatten_dict(tree)
    selected = {p: v for p, v in flat.items()
                if p[:2] == ("params", "params") and p[-2] in PORTS and p[-1] == "kernel"}
    if len(selected) != 7:
        raise ValueError(f"Expected three scan groups and one sparse MLP writer: {list(selected)}")
    sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    items = {p: jax.ShapeDtypeStruct(v.shape, v.dtype) for p, v in selected.items()}
    restore_args = {p: ocp.ArrayRestoreArgs(sharding=sharding, global_shape=v.shape,
                                          dtype=v.dtype) for p, v in selected.items()}
    restored = checkpointer.restore(args.checkpoint, args=ocp.args.PyTreeRestore(
        item=unflatten_dict(items), restore_args=unflatten_dict(restore_args), partial_restore=True))
    arrays = {"/".join(p): np.asarray(v, dtype=np.float32)
              for p, v in flatten_dict(restored).items()}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "down-projections.npz", **arrays)
    manifest = {"checkpoint": args.checkpoint, "backend": "cpu", "arrays": {
        name: {"shape": list(a.shape), "finite": bool(np.isfinite(a).all())}
        for name, a in arrays.items()}}
    (out / "weights-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == "__main__":
    main()
