"""Keep old terminal state and cleanup ownership outside the current run."""
def same_attempt(record, manifest):
 expected=manifest.get('attempt_id')
 if not isinstance(expected,str) or not expected:raise ValueError('A nonempty attempt_id is required')
 return record.get('attempt_id')==expected and record.get('source_commit')==manifest['commit']
def owned_resources(records, manifest):
 return [x for x in records if x.get('attempt_id')==manifest['attempt_id'] and x.get('owner_task')==manifest['task']]
