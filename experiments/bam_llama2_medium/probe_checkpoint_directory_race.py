"""Isolated GCS/TF directory-cache probe; never touches a RUN checkpoint.

Run in the training environment (CPU only). The sole argument is a GCS prefix
ending in /diagnostics/checkpoint_directory_probe. Each invocation owns a new
UUID child and removes only the objects it created. Prints raw observations.
"""

import importlib.metadata
import json
import os
import sys
import uuid

from etils import epath
from google.cloud import storage
import tensorflow as tf


def main():
    root = sys.argv[1].rstrip('/')
    assert root.startswith('gs://') and root.endswith(
        '/diagnostics/checkpoint_directory_probe')
    uri = root + '/' + uuid.uuid4().hex
    bucket_name, prefix = uri[5:].split('/', 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    directory = epath.Path(uri)
    cursor = directory / 'skip_file_and_step.json'
    result = {
        'uri': uri,
        'versions': {name: importlib.metadata.version(name)
                     for name in ('tensorflow', 'etils')},
        'stat_cache_max_age': os.environ.get('GCS_STAT_CACHE_MAX_AGE', 'default'),
    }
    try:
        # GCS permits a child object without a directory-marker object, as in
        # record_file_and_step immediately after an async Orbax save returns.
        with cursor.open('w') as handle:
            handle.write('{"checkpoint_step": 0}')
        result['objects_before'] = [b.name for b in client.list_blobs(
            bucket_name, prefix=prefix + '/')]
        result['exists_before'] = directory.exists()
        directory.rmtree()
        result['objects_after_delete'] = [b.name for b in client.list_blobs(
            bucket_name, prefix=prefix + '/')]
        result['tf_exists_after_delete'] = directory.exists()
        try:
            directory.mkdir(parents=True, exist_ok=False)
            result['mkdir'] = 'OK'
        except FileExistsError as error:
            result['mkdir'] = f'{type(error).__name__}: {error}'
    finally:
        # Exact names only; these are the only two objects this probe can create.
        for name in (prefix + '/skip_file_and_step.json', prefix + '/'):
            blob = bucket.blob(name)
            if blob.exists():
                blob.reload()
                blob.delete(if_generation_match=blob.generation)
        result['remaining_objects'] = [b.name for b in client.list_blobs(
            bucket_name, prefix=prefix + '/')]
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
