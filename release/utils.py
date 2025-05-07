

import json
from pathlib import Path
from typing import Dict, Any, List, Union
def read_jsonl(path: Union[Path, str], replace_newlines: bool = False) -> List[Dict[str, Any]]:
    """
    Read jsonlines from `path`, supporting .zip and .gz files.
    """
    # handle regular files
    with open(path) as infile:
        if not replace_newlines:
            return [json.loads(line) for line in infile if line]
        else:
            return [json.loads(line.replace("\\n", " ")) for line in infile if line]

def write_jsonl(data: List, path: Union[Path, str]):
    """
    Write jsonlines to `path`
    """
    with open(path, 'w') as outfile:
        for i, row in enumerate(data):
            outfile.write((i>0)*'\n' + json.dumps(row))
