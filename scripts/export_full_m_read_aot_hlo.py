"""CPU-host AOT only: export exact target-topology optimized HLO alongside executable."""
import gzip,json,os,sys
from pathlib import Path
from absl import app
import train_compile
original_save=train_compile.save_compiled

def save_and_export(compiled,destination):
    original_save(compiled,destination)
    stem=Path(destination)
    with gzip.open(str(stem)+'.hlo.txt.gz','wt') as f:
        f.write(compiled.as_text())
    metadata={'argv':sys.argv,'compiled':str(stem)}
    for name in ('memory_analysis','cost_analysis'):
        try:
            metadata[name]=str(getattr(compiled,name)())
        except Exception as exc:
            metadata[name+'_unavailable']=str(exc)
    Path(str(stem)+'.analysis.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('HLO_EXPORT_OK '+str(stem)+'.hlo.txt.gz',flush=True)

train_compile.save_compiled=save_and_export
if __name__=='__main__':
    app.run(train_compile.main)
