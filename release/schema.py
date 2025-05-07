GROUP_TYPES = [
  "social group",
  "political group",
  "political institution",
  "organization, public institution, or collective actor",
  "implicit social group reference",
]

# convert to IOB2 scheme
scheme = ['O'] + ['I-'+t for t in GROUP_TYPES] + ['B-'+t for t in GROUP_TYPES]

ID2LABEL = {i: l for i, l in enumerate(scheme)}
LABEL2ID = {l: i for i, l in enumerate(scheme)}
