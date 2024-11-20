
import os
import pandas as pd


data_path = os.path.join('..', 'data', 'exdata', 'cap')


# (down)load the data
fp = os.path.join(data_path, 'uk_manifestos_v1.csv')
if not os.path.exists(fp):
    # note: Unfortunately, the CAP folks have edited the orginal file named "uk_manifestos_v1.csv" after we downloaded. 
    #       If you now go to https://comparativeagendas.s3.amazonaws.com/ (2024-08-21) and search for "uk_manifestos_v1",
    #        you'll see that there are now several files ("uk_manifestos_v1_1.csv", "uk_manifestos_v1_1_1.csv", etc.).
    #       We assume that the file we downloaded is now called "uk_manifestos_v1_1.csv"
    #       But to make sure that we have the correct file for replication, we have stored a copy of the file we 
    #        originally downloaded in the replication materials (see `fp`).
    url = 'https://comparativeagendas.s3.amazonaws.com/datasetfiles/uk_manifestos_v1_1.csv'
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    df = pd.read_csv(url, encoding='ISO-8859-1')
    df.to_csv(fp, encoding='utf-8', index=False)
else:
    df = pd.read_csv(fp)


# clean

## rename text column
df = df.rename(columns={'description': 'text'})

## clean text
df['text'] = df.text.str.strip()

## drop missings
df = df[df.text != '']
df = df[~df.text.isna()]


# Note: the data contain LibDem manifestos (1983-2015) and one UKIP manifesto (2015)
#  but our target time-series data records measures for the parties only for/since 2015
df[df.filter_ld==1].year.value_counts()
df[df.filter_ukip==1].year.value_counts()

#  Hence, we discard them and focus only on sentences from Labour and Conservative manifestsos
df = df[df.filter_ld==0]
df = df[df.filter_ukip==0]
df['party'] = df.filter_con.apply(lambda x: 'con' if x==1 else 'lab')
# df[['year', 'filter_con', 'filter_lab', 'party']].value_counts(sort=False)


# clean texts

# note lot's of weird characters
set([char for t in df.text for char in t])

replace_these = [
    '\\x85',
    '\\x91',
    '\\x92',
    '\\x93',
    '\\x94',
    '\\x95',
    '\\x96',
    '\\x97',
    '\\xa0',
]

df.loc[:, 'text'] = df.text.str.replace(r'[%s]+' % ''.join(replace_these), ' ', regex=True).str.replace(r'\s+', ' ', regex=True)


# source: codebook, page 9 (https://comparativeagendas.s3.amazonaws.com/codebookfiles/uk_manifestos_codebook_v1.pdf)
id2label = {
    0: 'Non-policy content',
    1: 'Macroeconomics',
    2: 'Civil Rights, Minority Issues, Immigration and Civil Liberties',
    3: 'Health',
    4: 'Agriculture',
    5: 'Labour and Employment',
    6: 'Education and Culture',
    7: 'Environment',
    8: 'Energy',
    9: 'Immigration',
    10: 'Transportation',
    12: 'Law, Crime and Family Issues',
    13: 'Social Welfare',
    14: 'Community Development, Planning and Housing Issues',
    15: 'Banking, Finance and Domestic Commerce',
    16: 'Defence',
    17: 'Space, Science, Technology and Communications',
    18: 'Foreign Trade',
    19: 'International Affairs and Foreign Aid',
    20: 'Government Operations',
    21: 'Public Lands, Water Management, Colonial and Territorial Issues',
}


label2id = {l: i for i, l in id2label.items()}


df['majortopic_label'] = df.majortopic.map(id2label)


df[['majortopic_label', 'majortopic']].value_counts(dropna=False).sort_index(axis=0)


# remove topic 23 (not in codebook)
df = df[~df.majortopic_label.isna()]


id2label_recoded = {
      ## 0: 'Non-policy content'
    1: 'Macroeconomics',
    2: 'Civil Rights, Minority Issues, Immigration and Civil Liberties',
    3: 'Health',
    4: 'Agriculture',
    5: 'Labour and Employment',
    6: 'Education and Culture',
    7: 'Environment',
      ## 8: 'Energy',
    9: 'Civil Rights, Minority Issues, Immigration and Civil Liberties', # originally 'Immigration',
    10: 'Transportation',
    12: 'Law, Crime and Family Issues',
    13: 'Social Welfare',
    14: 'Community Development, Planning and Housing Issues',
      ## 15: 'Banking, Finance and Domestic Commerce',
      ## 16: 'Defence',
      ## 17: 'Space, Science, Technology and Communications',
      ## 18: 'Foreign Trade',
      ## 19: 'International Affairs and Foreign Aid',
      ## 20: 'Government Operations',
      ## 21: 'Public Lands, Water Management, Colonial and Territorial Issues',
}

def recode_cap(code):
    return id2label_recoded[code] if code in id2label_recoded.keys() else 'other'

df['majortopic_recoded_label'] = df.majortopic.apply(recode_cap)


# subset to relevant columns
df = df[['id', 'year', 'party', 'text', 'majortopic', 'majortopic_label', 'majortopic_recoded_label']].reset_index(drop=True)


fp = os.path.join(data_path, 'cap_codes_uk-manifesto_sentences_con+lab.csv')
df.to_csv(fp, index=False)


