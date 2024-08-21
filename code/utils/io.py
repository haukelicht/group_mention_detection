import json

def read_label_config(fp, outside_label=0):
    with open(fp, 'r') as f: 
        label_config = json.load(f)
    cat2code = {'O': outside_label}
    for i, l in enumerate(label_config):
        cat2code['I-'+l['text']] = int(i+1)
        cat2code['B-'+l['text']] = int(i+1+len(label_config))
    return cat2code