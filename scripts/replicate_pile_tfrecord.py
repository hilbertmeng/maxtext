#!/usr/bin/env python3
"""Copy completed TruePile shards, audit every replica, then publish readiness."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import struct
import subprocess

import google_crc32c
from google.cloud import storage
from tensorflow.core.example.example_pb2 import Example


def split_uri(uri):
    bucket, prefix = uri.removeprefix('gs://').rstrip('/').split('/', 1)
    return bucket, prefix


def masked_crc(data):
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xffffffff


def audit(client, uri, manifest):
    bucket_name, prefix = split_uri(uri)
    bucket = client.bucket(bucket_name)
    actual = {blob.name.rsplit('/', 1)[-1]: blob for blob in
              client.list_blobs(bucket, prefix=prefix+'/pile.tfrecord.b')}
    expected = {row['shard']: row for row in manifest['shards']}
    if set(actual) != set(expected):
        raise ValueError(f'{uri}: shard inventory differs: missing={set(expected)-set(actual)}, extra={set(actual)-set(expected)}')

    def verify(item):
        name, row = item
        blob = actual[name]
        if int(blob.size) != row['bytes'] or blob.crc32c != row['crc32c']:
            raise ValueError(f'{uri}/{name}: size/CRC mismatch')
        # The range is small, but the entire object is checked against the builder's CRC above.
        data = blob.download_as_bytes(start=0, end=65535, checksum=None,
                                      if_generation_match=blob.generation)
        length, header_crc = struct.unpack('<QI', data[:12])
        if len(data) < length+16:
            raise ValueError(f'{uri}/{name}: first record exceeds audit range')
        payload = data[12:12+length]
        payload_crc = struct.unpack('<I', data[12+length:16+length])[0]
        if masked_crc(data[:8]) != header_crc or masked_crc(payload) != payload_crc:
            raise ValueError(f'{uri}/{name}: TFRecord framing CRC mismatch')
        example = Example.FromString(payload)
        tokens = example.features.feature['input_ids'].int64_list.value
        if len(tokens) != manifest['record_length']:
            raise ValueError(f'{uri}/{name}: {len(tokens)} tokens, expected {manifest["record_length"]}')
        return row['bytes']
    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        sizes = list(executor.map(verify, expected.items()))
    return dict(uri=uri, shards=len(sizes), bytes=sum(sizes),
                samples=manifest['samples'], record_length=manifest['record_length'],
                every_shard_size_crc_and_record_length_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', action='append', required=True)
    parser.add_argument('--audit-output', required=True)
    args = parser.parse_args()
    client = storage.Client()
    bucket, prefix = split_uri(args.source)
    manifest_bytes = client.bucket(bucket).blob(prefix+'/manifest.json').download_as_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest['record_length'] != manifest['seq_length']+1 or
        manifest['samples'] != manifest['steps']*manifest['global_batch_size'] or
        sum(row['samples'] for row in manifest['shards']) != manifest['samples']):
        raise ValueError('inconsistent source coverage')
    results = [audit(client, args.source, manifest)]
    print('SOURCE_AUDITED', json.dumps(results[0]), flush=True)

    def replicate(target):
        if target.rstrip('/') == args.source.rstrip('/'):
            raise ValueError('target is source')
        subprocess.run(['gcloud', 'storage', 'cp', args.source.rstrip('/')+'/pile.tfrecord.b*',
                        target.rstrip('/')+'/'], check=True)
        result = audit(client, target, manifest)
        bucket, prefix = split_uri(target)
        client.bucket(bucket).blob(prefix+'/manifest.json').upload_from_string(
            manifest_bytes, content_type='application/json')
        print('REPLICA_COMMITTED', json.dumps(result), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        results.extend(executor.map(replicate, args.target))
    Path(args.audit_output).write_text(json.dumps(results, indent=2)+'\n')
    print('ALL_REPLICAS_READY', flush=True)


if __name__ == '__main__':
    main()
