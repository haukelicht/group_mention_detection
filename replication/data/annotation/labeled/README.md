# Labeled human-annotated sentences

The files in this folder contain the labels aggregated from our two coders' group mention annotations.

Each file is a JSONlines file, that is a text file with one JSON dictionary per line.

An exemplary line looks like this:

```json
Die Dateien sind JSONlines files. Jede Zeile sieht quasi so aus:

{'id': '829ac29cd9304a66265e3ea830a505e3',
 'text': 'Seit 150 Jahren machen wir Politik für eine bessere Gesellschaft .',
 'tokens': ['Seit',
  '150',
  'Jahren',
  'machen',
  'wir',
  'Politik',
  'für',
  'eine',
  'bessere',
  'Gesellschaft',
  '.'],
 'annotations': {'emarie': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
 'labels': {'BSCModel': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
 'metadata': {'sentence_id': '41320.000.2013.1.1-2520-1',
  'split_': 'smarie',
  'job': 'group-mentions-annotation-de-manifestos-round-01'}
}
```

Note that the dictionary records 

- the text in pre-tokenized format,
- annotations and labels at the _token_ level (a dictionary of lists, one per annotator),

Annotations are in field `"annotations"` and map annotator IDs to their token-level annotations.
**Labels** are in field `"labels"` and `"BSCModel"` are the Bayesian Sequence Combination (BSC) model-based aggregate labels.

## Converting numeric label IDs to label indicators and entity types

The data in `"annotations"` and `"labels"` are numeric label IDs that map onto our **group types**:

- social group
- political group
- political institution
- organization, public institution, or collective actor
- implicit social group reference

To convert them to text labels, use this python code:

```python
# get list of entity types
entity_types = [
  "social group",
  "political group",
  "political institution",
  "organization, public institution, or collective actor",
  "implicit social group reference",
  "unsure",
]
# convert to IOB2 scheme
scheme = ['O'] + ['I-'+t for t in types] + ['B-'+t for t in types]
# map label type indicators to label IDs
label2id = {l: i for i, l in enumerate(scheme)}
# and vice versa
id2label = {i: l for i, l in enumerate(scheme)}
```
