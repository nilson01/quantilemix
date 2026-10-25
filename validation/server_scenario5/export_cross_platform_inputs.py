import hashlib
import json
from pathlib import Path
import sys
from validation import paper_scenario5 as paper

result = json.loads(Path(sys.argv[1]).read_text())
identities = {(r['variant'], r['cell'], r['repetition']): r for r in result['records']}
keys = [('released_ncor2', 2, 704), ('released_ncor2', 5, 335),
        ('released_ncor2', 7, 825), ('two_causal_ncor1', 3, 443),
        ('two_causal_ncor1', 6, 978)]
rows = []
for key in keys:
    x, y = paper.generate(*key, mode='study')
    if paper.input_hash(x, y) != identities[key]['input_sha256']:
        raise SystemExit(f'server input differs from saved study: {key}')
    rows.append({'identity': key, 'input_sha256': paper.input_hash(x, y),
                 'x_sha256': hashlib.sha256(x.astype('<i8').tobytes()).hexdigest(),
                 'y_sha256': hashlib.sha256(y.astype('<f8').tobytes()).hexdigest(),
                 'y': y.tolist()})
Path(sys.argv[2]).write_text(json.dumps(rows, allow_nan=False))
