#!/usr/bin/env python3
"""Build true 4097-token Pile TFRecords from Pythia mmap/index mappings.

The released 2048 sample map describes consecutive 2049-token windows over a
shuffled-document stream.  Every second boundary therefore describes a true
4097-token window without joining already-shuffled training examples.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time

import numpy as np


DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


class MMapDocumentDataset:
    """Minimal reader for the Megatron MMIDIDX format."""

    def __init__(self, prefix: str):
        idx_path = prefix + ".idx"
        with open(idx_path, "rb") as stream:
            if stream.read(9) != b"MMIDIDX\x00\x00":
                raise ValueError(f"bad mmap index magic: {idx_path}")
            version = struct.unpack("<Q", stream.read(8))[0]
            if version != 1:
                raise ValueError(f"unsupported mmap index version: {version}")
            dtype_code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = np.dtype(DTYPES[dtype_code])
            self.length = struct.unpack("<Q", stream.read(8))[0]
            self.doc_count = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

        self._idx_mmap = np.memmap(idx_path, mode="r", order="C")
        self._idx_mmap._mmap.madvise(mmap.MADV_RANDOM)
        buffer = memoryview(self._idx_mmap)
        self.sizes = np.frombuffer(buffer, dtype=np.int32, count=self.length, offset=offset)
        self.pointers = np.frombuffer(
            buffer,
            dtype=np.int64,
            count=self.length,
            offset=offset + self.sizes.nbytes,
        )
        self._bin_mmap = np.memmap(prefix + ".bin", mode="r", order="C")
        # Samples follow a shuffled document index, so accesses into the 619 GiB
        # token file are random.  Disable kernel read-ahead; otherwise every tiny
        # document read pulls a large unused window and saturates the local SSDs.
        self._bin_mmap._mmap.madvise(mmap.MADV_RANDOM)
        self._bin_buffer = memoryview(self._bin_mmap)

    def get(self, document: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        size = int(self.sizes[document])
        if length is None:
            length = size - offset
        pointer = int(self.pointers[document]) + offset * self.dtype.itemsize
        return np.frombuffer(self._bin_buffer, dtype=self.dtype, count=length, offset=pointer)


def materialize_sample(
    dataset: MMapDocumentDataset,
    doc_idx: np.ndarray,
    sample_idx: np.ndarray,
    sample_id: int,
    boundary_stride: int,
    expected_length: int,
) -> np.ndarray:
    first = sample_idx[boundary_stride * sample_id]
    last = sample_idx[boundary_stride * (sample_id + 1)]
    doc_first, offset_first = int(first[0]), int(first[1])
    doc_last, offset_last = int(last[0]), int(last[1])

    if doc_first == doc_last:
        sample = dataset.get(
            int(doc_idx[doc_first]),
            offset=offset_first,
            length=offset_last - offset_first + 1,
        )
    else:
        pieces = [dataset.get(int(doc_idx[doc_first]), offset=offset_first)]
        pieces.extend(dataset.get(int(doc_idx[pos])) for pos in range(doc_first + 1, doc_last))
        pieces.append(dataset.get(int(doc_idx[doc_last]), length=offset_last + 1))
        sample = np.concatenate(pieces)

    if sample.size != expected_length:
        raise ValueError(
            f"sample {sample_id} has {sample.size} tokens; expected {expected_length}"
        )
    return sample


def run(command: list[str], *, attempts: int = 4) -> None:
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            return
        if attempt == attempts:
            raise subprocess.CalledProcessError(result.returncode, command)
        time.sleep(5 * attempt)


def gcs_exists(uri: str) -> bool:
    return subprocess.run(
        ["gcloud", "storage", "objects", "describe", uri],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def write_shard(task: tuple) -> dict:
    (
        shard_index,
        args_dict,
        selected_path,
    ) = task
    # Importing TensorFlow inside each worker avoids initializing it before fork.
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    args = argparse.Namespace(**args_dict)
    start_step = shard_index * args.steps_per_shard
    steps = min(args.steps_per_shard, args.steps - start_step)
    samples = steps * args.global_batch_size
    sample_offset = start_step * args.global_batch_size
    name = f"pile.tfrecord.b{start_step}"
    destination = f"{args.output_uri.rstrip('/')}/{name}"
    if gcs_exists(destination):
        return {"shard": name, "status": "exists", "samples": samples}

    dataset = MMapDocumentDataset(args.data_prefix)
    doc_idx = np.load(args.doc_idx, mmap_mode="r")
    sample_idx = np.load(args.sample_idx, mmap_mode="r")
    selected = np.load(selected_path, mmap_mode="r")

    output_dir = Path(args.work_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".{name}.{os.getpid()}.tmp"
    final = output_dir / name
    options = tf.io.TFRecordOptions(compression_type="")
    with tf.io.TFRecordWriter(str(temporary), options=options) as writer:
        for local_index in range(samples):
            sample_id = int(selected[sample_offset + local_index])
            tokens = materialize_sample(
                dataset,
                doc_idx,
                sample_idx,
                sample_id,
                args.boundary_stride,
                args.seq_length + 1,
            )
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "input_ids": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=tokens)
                        )
                    }
                )
            )
            writer.write(example.SerializeToString())
    os.replace(temporary, final)
    size = final.stat().st_size
    run(["gcloud", "storage", "cp", str(final), destination])
    final.unlink()
    return {"shard": name, "status": "uploaded", "samples": samples, "bytes": size}


def build_selected_ids(args: argparse.Namespace, total_4k_samples: int) -> tuple[Path, str]:
    selected_path = Path(args.work_dir) / "selected_sample_ids.npy"
    metadata_path = Path(args.work_dir) / "selected_sample_ids.json"
    required = args.steps * args.global_batch_size
    if selected_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("seed") == args.seed
            and metadata.get("population") == total_4k_samples
            and metadata.get("required") == required
        ):
            digest = hashlib.sha256(selected_path.read_bytes()).hexdigest()
            if digest == metadata.get("sha256"):
                return selected_path, digest

    rng = np.random.RandomState(args.seed)
    permutation = np.arange(total_4k_samples, dtype=np.uint32)
    rng.shuffle(permutation)
    np.save(selected_path, permutation[:required], allow_pickle=False)
    del permutation
    digest = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    metadata_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "population": total_4k_samples,
                "required": required,
                "sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return selected_path, digest


def validate_inputs(args: argparse.Namespace) -> tuple[int, np.dtype]:
    dataset = MMapDocumentDataset(args.data_prefix)
    doc_idx = np.load(args.doc_idx, mmap_mode="r")
    sample_idx = np.load(args.sample_idx, mmap_mode="r")
    if sample_idx.ndim != 2 or sample_idx.shape[1] != 2:
        raise ValueError(f"unexpected sample_idx shape: {sample_idx.shape}")
    if args.seq_length % args.base_seq_length:
        raise ValueError("seq_length must be a multiple of base_seq_length")
    boundary_stride = args.seq_length // args.base_seq_length
    args.boundary_stride = boundary_stride
    total_samples = (sample_idx.shape[0] - 1) // boundary_stride
    required = args.steps * args.global_batch_size
    if required > total_samples:
        raise ValueError(f"need {required} samples but only {total_samples} are available")
    if int(doc_idx.max()) >= dataset.length:
        raise ValueError("doc_idx references a document outside the mmap dataset")

    probes = np.linspace(0, total_samples - 1, num=17, dtype=np.int64)
    for sample_id in probes:
        materialize_sample(
            dataset,
            doc_idx,
            sample_idx,
            int(sample_id),
            boundary_stride,
            args.seq_length + 1,
        )
    return total_samples, dataset.dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-prefix", required=True)
    parser.add_argument("--doc-idx", required=True)
    parser.add_argument("--sample-idx", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--base-seq-length", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=13_500)
    parser.add_argument("--steps-per-shard", type=int, default=500)
    parser.add_argument("--seed", type=int, default=9876)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    total_samples, token_dtype = validate_inputs(args)
    print(
        f"validated source: dtype={token_dtype} available_4096_samples={total_samples} "
        f"required={args.steps * args.global_batch_size}",
        flush=True,
    )
    if args.validate_only:
        return

    selected_path, selection_sha = build_selected_ids(args, total_samples)
    manifest = {
        "schema_version": 1,
        "source_data_prefix": args.data_prefix,
        "source_doc_idx": args.doc_idx,
        "source_sample_idx": args.sample_idx,
        "seq_length": args.seq_length,
        "record_length": args.seq_length + 1,
        "base_seq_length": args.base_seq_length,
        "boundary_stride": args.boundary_stride,
        "global_batch_size": args.global_batch_size,
        "steps": args.steps,
        "steps_per_shard": args.steps_per_shard,
        "seed": args.seed,
        "selection_sha256": selection_sha,
    }
    local_manifest = Path(args.work_dir) / "manifest.json"
    local_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    run(
        [
            "gcloud",
            "storage",
            "cp",
            str(local_manifest),
            f"{args.output_uri.rstrip('/')}/manifest.json",
        ]
    )

    shard_count = (args.steps + args.steps_per_shard - 1) // args.steps_per_shard
    args_dict = vars(args).copy()
    tasks = [(index, args_dict, str(selected_path)) for index in range(shard_count)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(write_shard, task): task[0] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
