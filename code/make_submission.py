"""Fill highlights and cover letter from results/numbers.json (run after make_numbers.py)."""
import json
import re
import sys
sys.path.insert(0, 'code')
N = json.load(open('results/numbers.json'))
from render import fmt  # noqa: E402  (render.py builds main.md on import; harmless)
for name in ('highlights', 'cover_letter'):
    t = open(f'submission/templates/{name}.md').read()
    out = re.sub(r'\{\{(\w+)\|(\w+)\}\}', lambda m: fmt(N[m.group(1)], m.group(2)), t)
    open(f'submission/{name}.md', 'w').write(out)
    if name == 'highlights':
        for line in out.strip().splitlines():
            assert len(line.lstrip('- ')) <= 85, (len(line), line)
print('highlights and cover letter written')
