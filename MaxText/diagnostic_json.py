"""Strict JSON output for diagnostic metadata, including NumPy scalars."""
import json
import numpy as np

def _scalar(value):
 if isinstance(value,np.generic):return value.item()
 raise TypeError(f'Unsupported diagnostic metadata: {type(value).__name__}')

def dumps(value,**kwargs):
 return json.dumps(value,default=_scalar,allow_nan=False,**kwargs)
