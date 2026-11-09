"""Current V5 kernel API, with no dispatch to historical implementations."""
from layers.bam_pallas_common import static_weight
from layers.bam_pallas_v8 import read_tm as read, write_tm as write
