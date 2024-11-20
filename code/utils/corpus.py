import json
import jsonlines
import re
import regex
import os
from tqdm import tqdm
import numpy as np
from random import sample
from collections import Counter

"""
Class labels for 1-type sequence annotation/tagging/labeling in IOB scheme

0 := Inside a span (all but the first token)
1 := Outside a span/spans
2 := Beginning (for token) of a span
"""
INSIDE_LABEL = int(0)
OUTSIDE_LABEL = int(1)
BEGIN_LABEL = int(2)

class Document(object):
    
    def __init__(self, id, text):
        assert isinstance(id, str)
        self.id = id
        assert isinstance(text, str)
        self.text = text
        self.tokens = self._ws_tokenize(text)
        self.n_tokens = len(self.tokens)
    
    # TODO: replace the white-space tokenizer with a more sophisticated one that preserved leading white spaces 
    
    # white space tokenizer
    @classmethod
    def _ws_tokenize(cls, doc):
        doc = re.sub("\x20+", u"\x20", doc)
        toks = doc.split(u"\x20")
        return toks

    def _jsonify(self):
        return json.dumps({k: self.__dict__[k] for k in ['id', 'text', 'tokens']})


class AnnotatedDocument(Document):
    def __init__(self, id, text, annotations, labels = None, outside_label = 1):
        
        super().__init__(id, text)
        
        # meta data
        self.outside_label = outside_label

        # data
        self.annotators = list(annotations.keys())
        self.annotations = annotations
        self.is_annotated = len(self.annotations) > 0
        self.n_annotations = len(self.annotations)
        self.is_labeled = labels is not None
        self.labels = dict() if not self.is_labeled else labels
        self.n_labels = len(self.labels)
        # generate string for print method
        self._generate_printable()
    
    @classmethod
    def from_tokens(cls, id: str, tokens: str, annotations: dict, outside_label: int, unite_pattern = None):
        if unite_pattern:
            text = ''
            for tok in tokens:
                text += tok if re.search(unite_pattern, tok) else u'\x20'+tok
        else:
            text = u"\x20".join(tokens)
        # TODO: handle case when labels are available
        doc = cls(id, text, annotations, None, outside_label)
        doc.tokens = tokens
        doc.n_tokens = len(tokens)
        return doc
    
    def clean_tokens(self, fun):
        self.tokens_cleaned_ = [fun(tok) for tok in self.tokens]
        return self.tokens_cleaned_

    def merge_document(self, doc):
        try:
            for annotator_id, annotation in doc.annotations.items():
                self.add_annotation(annotator_id, annotation, regenerate_printable = False)
        except:
            print(doc.id)
        # TODO: merge labels
        self._generate_printable()

    def add_annotation(self, id, annotation, regenerate_printable = True):
        """Add an annotation to a sequence"""
        if id in self.annotations.keys():
            return
        assert len(annotation) == self.n_tokens, "annotation too " + "long" if len(annotation) > self.n_tokens else "short"
        self.annotations[id] = annotation
        self.annotators.append(id)
        self.n_annotations += 1
        self.is_annotated = True
        if regenerate_printable:
            self._generate_printable()

    def remove_annotation(self, id):
        assert id in self.annotators, print(self.id, id)
        self.annotations.pop(id)
        self.annotators = [annotator for annotator in self.annotators if annotator != id]
        self.n_annotations -= 1
        self.is_annotated = self.n_annotations > 0
        self._generate_printable()

    def add_labels(self, labels, source, regenerate_printable = True):
        """Add labels (ground truth) to an annotated sequence"""
        assert len(labels) == self.n_tokens, "too " + "many labels" if len(labels) > self.n_tokens else "few labels"
        self.labels[source] = labels
        self.n_labels += 1
        if regenerate_printable:
            self._generate_printable()
    
    def remove_labels(self, source):
        assert source in self.labels, print(self.id, source)
        self.labels.pop(source)
        self.n_labels -= 1
        self._generate_printable()
        
    def segment_document(self, regex = None):
        docs = list()
        if not isinstance(regex, re.Pattern):
            regex = re.compile(regex)
        
        # return empty list if pattern not in doc's text
        if not re.search(regex, self.text):
            return docs
        
        # segment by regex pattern 
        prev = 0
        splitted_at = [""]
        for span in re.finditer(regex, self.text):
            splitted_at.append( self.text[span.end()-2:span.end()+1] )
            docs.append( self.text[prev:span.end()].strip() )
            prev = span.end()
        if prev != len(self.text):
            docs.append( self.text[prev:len(self.text)].strip() )
        
        # return if segmentation yields only one doc
        # note: this happens only if the pattern occurs only once at the end of the doc's text
        if len(docs) == 1:
            return list()
        
        # split annotations (and cleaned tokens, if any)
        out = list()
        first_tok = 0
        has_cleaned_tokens = hasattr(self, "tokens_cleaned_")
        for i, doc in enumerate(docs):
            # need to get annotation of preceding when    split does not contain white space
            adjust_ = False if i == 0 or bool(re.match(r"^\x20", splitted_at[i])) else True
            last_tok = doc.count(u"\x20")+first_tok+int(i == 0)
            annotations = dict()
            for id, annotation in self.annotations.items():
                annotations[id] = annotation[(first_tok-int(adjust_)):last_tok]
            labels = dict()
            for id, labs in self.labels.items():
                labels[id] = labs[(first_tok-int(adjust_)):last_tok]
            # create IDs and AnnotatedDocument instances
            id_ = self.id + "-s" + str(i)
            doc = AnnotatedDocument(id_, doc, annotations, labels, self.outside_label)
            if has_cleaned_tokens:
                doc.tokens_cleaned_ = self.tokens_cleaned_[(first_tok-int(adjust_)):last_tok]
            out.append( (id_, doc))
            first_tok = last_tok-1
        
        # overwrite document attributes with data for first segment
        _, doc = out.pop(0)
        self.text = doc.text
        self.tokens = doc.tokens
        self.n_tokens = len(doc.tokens)
        if has_cleaned_tokens:
            self.tokens_cleaned_ = doc.tokens_cleaned_
        self.annotations = doc.annotations
        self.labels = doc.labels
        
        # return the rest
        return out

    def _jsonify(self):
        out = {k: self.__dict__[k] for k in ['id', 'text', 'tokens', 'annotations', 'labels']}
        out['annotations'] = {k: v.tolist() for k, v in out['annotations'].items()}
        out['labels'] = {k: v.tolist() for k, v in out['labels'].items()}
        if len(out['labels']) == 0:
            out['labels'] = None
        return json.dumps(out)

    def _generate_printable(self):
        printable = [f"\x1b[1m{self.id}\x1b[0m\n'{self.text}'"]
        if self.n_annotations > 0 or self.n_labels > 0:
            e = [len(tok) for tok in self.tokens]
            s = [0] + e[:-1] 
            s, e = np.asarray(s), np.asarray(e)
            s = s.cumsum() + np.arange(self.n_tokens)
            e = e.cumsum() + np.arange(self.n_tokens)
            e[-1] -= 1
            n_ = len(self.text)
            # add annotations (if any)
            for i, (id, a) in enumerate(self.annotations.items()):
                places = [" "]*n_
                for j, v in enumerate(a):
                    if v == self.outside_label:
                        continue
                    places[s[j]] = "\x1b[44m "
                    places[e[j]] = "\x1b[49m "
                printable.append(" " + "".join(places) + f"\t({id})" )
            # add labels (if any)
            for i, (src, a) in enumerate(self.labels.items()):
                places = [" "]*n_
                for j, v in enumerate(a):
                    if v == self.outside_label:
                        continue
                    places[s[j]] = "\033[42m "
                    places[e[j]] = "\033[49m "
                printable.append(" " + "".join(places) + f"\t[{src}]" )
            
            self._printable = "\n".join(printable)
        else:
            self._printable = printable[0]

    def __repr__(self):
        txt = self.tokens[slice(0, min(4, self.n_tokens))] 
        return f"AnnotatedDocument('{' '.join(txt)} ...')"

    def __str__(self):
        return self._printable

    def __len__(self):
        return self.n_tokens


class SequenceCorpus(object):
    
    def __init__(self):
        self.ndocs = None
        self.vocab = Counter()
        self.docs = list()
        self.ndocs = len(self.docs)
        self.doc_ids = list()
        self.doc_id2idx = dict()
        self.doc_idx2id = dict()
    
    def print_examples(self, k):
        if k > self.ndocs:
            raise ValueError(f"`k` cannot exceed corpus size ({self.ndocs})")
        for doc in sample(self.docs, k):
            print(doc)

    def add_document(self, doc):
        self.vocab.update( doc.tokens )
        self.doc_ids.append(doc.id)
        self.doc_id2idx[doc.id] = self.ndocs
        self.doc_idx2id[self.ndocs] = doc.id
        self.docs.append(doc)
        self.ndocs += 1
        # update vocabulary
        self._update_vocab()
    
    def remove_documents(self, doc_ids: list):
        self.docs = [doc for doc in self.docs if doc.id not in doc_ids]
        self.ndocs = len(self.docs)
        self.doc_ids = [doc.id for doc in self.docs]
        self.doc_id2idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        self.doc_idx2id = {i: doc_id for i, doc_id in enumerate(self.doc_ids)}
        self._update_vocab()

    def reset_doc_ids(self, new_ids):
        assert isinstance(new_ids, dict), "Warning"
        old_ids = set(new_ids.keys())
        for idx, doc_id in self.doc_idx2id.items():
            if doc_id in old_ids:
                self.doc_ids[idx] = new_ids[doc_id]
                self.doc_idx2id[idx] = new_ids[doc_id]
                self.docs[idx].id = new_ids[doc_id]
                self.docs[idx]._generate_printable()
        self.doc_id2idx = {doc_id: idx for idx, doc_id in self.doc_idx2id.items()}

    def _update_vocab(self):
        vocab = Counter()
        for doc in self.docs:
            vocab.update(doc.tokens)
        self.vocab = vocab
    

class AnnotatedSequenceCorpus(SequenceCorpus):

    def __init__(self, label_map, n_types=None):
        """

        :param label_map: dictionary mapping labels (keys) to integer codes (values)
        """
        # def __init__(self, outside_label, beginning_labels, inside_labels):
        super().__init__()
        self.annotators = Counter()
        self.annotator_label_counts = dict()
        self.label_map = label_map
        self.label_map_inv = {c: l for l, c in label_map.items()}
        self.inside_labels = self._get_label_indexes('I')
        self.outside_label = self._get_label_indexes('O', 0)
        self.beginning_labels = self._get_label_indexes('B')
        self.n_types = n_types if n_types is not None else len(self.beginning_labels)

    def _count_annotator_labels(self):
        counts = {id: Counter() for id in list(self.annotators.keys())}
        for doc in self.docs:
            for id, annotation in doc.annotations.items():
                counts[id].update(annotation.tolist())
        return counts

    def save_as_jsonlines(self, fp, **kwargs):
        with open(fp, mode='w', **kwargs) as writer:

            for doc in self.docs:
                writer.write(doc._jsonify()+'\n')
    
    def merge_annotated_corpus(self, annotated_corpus):
        for doc in annotated_corpus.docs:
            if doc.id in self.doc_ids:
                self.docs[self.doc_id2idx[doc.id]].merge_document(doc)
            else:
                self.add_document(doc, skip_unannotated=False)
        self._update_annotators()
        self.annotator_label_counts = self._count_annotator_labels()

    def merge_gold_corpus(self, gold_corpus, label_name = None):
        src = list(gold_corpus.annotators.keys())[0]
        label_name = src if label_name is None else label_name
        for doc in gold_corpus.docs:
            if doc.id in self.doc_ids:
                self.docs[self.doc_id2idx[doc.id]].add_labels(doc.annotations[src], label_name)
            else:
                # TODO: this is not great (it false to write to .labels, and it does not accept the `label_name` argument)
                self.add_document(doc, skip_unannotated=False)

    def add_document(self, doc, skip_unannotated = False):
        if skip_unannotated and doc.n_annotations == 0:
            return
        self.vocab.update( doc.tokens )
        self.doc_ids.append(doc.id)
        self.doc_id2idx[doc.id] = len(self.docs)
        self.doc_idx2id[len(self.docs)] = doc.id
        self.docs.append(doc)
        self.annotators.update(doc.annotators)
        for annotator_id, annotation in doc.annotations.items():
            if annotator_id not in self.annotator_label_counts.keys():
                self.annotator_label_counts[annotator_id] = Counter()
            self.annotator_label_counts[annotator_id].update(annotation.tolist())
        self.ndocs += 1
        self._update_vocab()

    def merge_annotations(self, doc_ids: list):
        assert len(doc_ids) > 1
        assert doc_ids[0] in self.doc_ids
        rm = list()
        for doc_id in doc_ids[1:]:
            if doc_id in self.doc_ids:
                self.docs[self.doc_id2idx[doc_ids[0]]].merge_document(self.docs[self.doc_id2idx[doc_id]])
                rm.append(doc_id)
        self.remove_documents(rm)

    def segment_documents(self, regex = None):
        if not isinstance(regex, re.Pattern):
            regex = re.compile(regex)
        new_docs = list()
        for i in range(self.ndocs):
            new_docs += self.docs[i].segment_document(regex)
        if len(new_docs) > 0:
            for _, doc in new_docs:
                self.add_document(doc)
        # update vocabulary
        self._update_vocab()

    def remove_documents(self, doc_ids: list):
        super().remove_documents(doc_ids)
        self._update_annotators()
        # update annotator label counts
        self.annotator_label_counts = self._count_annotator_labels()

    def remove_annotators(self, annotator_ids):
        annotator_ids = [annotator_id for annotator_id in annotator_ids if annotator_id in self.annotators.keys()]
        if len(annotator_ids) == 0:
            return
        
        # remove from counter
        for annotator_id in annotator_ids:
            self.annotators.pop(annotator_id)
            self.annotator_label_counts.pop(annotator_id)

        # remove from docs
        for i in range(self.ndocs):
            if not any([annotator_id in annotator_ids for annotator_id in self.docs[i].annotators]): 
                continue
            for annotator_id in annotator_ids:
                if annotator_id in self.docs[i].annotators:
                    self.docs[i].remove_annotation(annotator_id)

    # update annotators
    def _update_annotators(self):
        annotators = Counter()
        for doc in self.docs:
            annotators.update(doc.annotators)
        self.annotators = annotators

    def _get_label_indexes(self, l, which = None):
        tmp = [idx for lab, idx in self.label_map.items() if lab[0] == l]
        if len(tmp) == 0:
            emsg = f'no label in `label_map` starts with \'{l}\''
            raise ValueError(emsg)
        if which is not None:
            tmp = tmp[which]
        return tmp


class DoccanoAnnotationsCorpus(AnnotatedSequenceCorpus):
    
    def __init__(self, label_map, n_types=None):
        super().__init__(label_map=label_map, n_types=n_types)
    
    def load_from_jsonlines(self, fp, annotator_id, replace_chars=None, verbose=False):
        assert os.path.exists(fp), IOError("file does not exist")
        assert len(self.docs) == 0, ValueError("There are already documents in the corpus. Use add_documents() method")
        with jsonlines.open(fp) as reader:
            for line in reader:
                try:
                    if replace_chars:
                        tmp = line['text']
                        for r, p in replace_chars.items():
                            tmp = re.sub(p, r, tmp)
                        line['text'] = tmp
                    doc_ = self._parse_doccano_output_line(line=line, annotator=annotator_id, verbose=verbose)
                except Exception as e:
                    print(line['id'], str(e))
                else:
                    self.add_document(doc = doc_)
        self.ndocs = len(self.docs)
    
    def _parse_doccano_output_line(self, line, annotator, verbose=False):
        # create zero 1d-array to record annotations (if any)
        annotations = np.zeros(len(line['text']), dtype=int)
        for a in line['label']: 
            annotations[a[0]:a[1]] = self.label_map['B-'+a[2]]
        
        # split the string at the pattern and preserve the matched characters
        pattern = r"([\p{Z}\p{Mc}\p{Ps}\p{Pe}\p{Pi}\p{Pf}\p{Po}])"
        tmp = regex.split(pattern, line['text'])
        tmp = [t for t in tmp if len(t) > 0]
        
        # induce the plurality annotation at the token level
        idxs = np.array([len(t) for t in tmp]).cumsum()
        toks, labs = list(), list()
        for i, (t, a) in enumerate(zip(tmp, np.split(annotations, idxs))):
            if regex.match(r'\s+', t):
                continue
            else: 
                toks.append(t)
            a_ = np.median(a)
            if a_ % 1 != 0:
                if verbose: print(f"Warning: annotation of token {i} in document {line['id']} disambiguated")
                a_ = a[0] if len(labs) > 0 and labs[-1] == a[0] else a[-1]
            labs.append(int(a_))
        
        # reset annotations inside a span
        labs = np.asarray(labs)
        labs[1:][np.all([labs[:-1] != 0, labs[:-1] == labs[1:]], axis = 0)] -= self.n_types
        
        # construct an AnnotatedDocument instance from the parsed data
        doc = AnnotatedDocument.from_tokens(
            id=line['id'],
            tokens=toks,
            annotations={annotator: labs},
            outside_label=self.label_map['O'],
        )

        return doc


class JsonlinesAnnotationsCorpus(AnnotatedSequenceCorpus):
    def __init__(self, label_map, n_types=None):
        super().__init__(label_map=label_map, n_types=n_types)
    
    def load_from_jsonlines(self, fp, skip=None, maxn=None, verbose=False):
        assert len(self.docs) == 0, ValueError("There are already documents in the corpus. Use add_documents() method")
        assert os.path.exists(fp), IOError("file does not exist")
        n_ = sum(1 for line in open(fp))
        with jsonlines.open(fp) as reader:
            cnt = 0
            for line in tqdm(reader, total=n_):
                cnt += 1
                if skip is not None and cnt < skip:
                    continue
                if maxn is not None and cnt > maxn:
                    break
                try:
                    doc_ = self._parse_json(line=line)
                except Exception as e:
                    if verbose: print(line['id'], str(e))
                else:
                    self.add_document(doc = doc_)
        self.ndocs = len(self.docs)
        self.doc_ids = [doc.id for doc in self.docs]
        self.doc_id2idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        self.doc_idx2id = {i: doc_id for i, doc_id in enumerate(self.doc_ids)}

    def _parse_json(self, line):
        doc = AnnotatedDocument.from_tokens(
            id=line['id'],
            tokens=line['tokens'],
            annotations={k: np.array(v, dtype=int) for k, v in line['annotations'].items()},
            outside_label=self.label_map['O'],
        )
        if line['labels'] is not None:
            doc.add_labels(np.array(line['labels']['GOLD'], dtype=int), 'GOLD')
        return doc

