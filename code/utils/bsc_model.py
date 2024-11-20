
import timeit

import os
import json

import numpy as np
import pandas as pd

from random import sample

from utils.bayesian_combination.bayesian_combination import BC
from utils.bayesian_combination.baselines.majority_voting import MajorityVoting as MV



class BSCModel(object):
	model_name = "BSCModel"

	def __init__(self, annotated_corpus, gold_labels = None, use_cleaned_tokens = True, **kwargs):
		# set corpus
		self.corpus = annotated_corpus
		self.corpus.annotator_id2idx_ = {a: i for i, a in enumerate(self.corpus.annotators)}

		self.tokens_field = "tokens_cleaned_" if use_cleaned_tokens else "tokens"
		self.C, self.doc_start, self.features = self._init_bsc_model_inputs()

		self.num_classes = len(self.corpus.beginning_labels) + len(self.corpus.inside_labels) + 1
		self.num_annotators = self.C.shape[1]
		self.num_docs = self.corpus.ndocs
		self.num_tokens = self.C.shape[0]

		if gold_labels is not None:
			self.gold = self._get_gold(gold_labels)

		# BSC model
		self.model = BC(
			L = self.num_classes,
			K = self.num_annotators,
			tagging_scheme = "IOB2",
			inside_labels = self.corpus.inside_labels,
			outside_label = self.corpus.outside_label,
			beginning_labels = self.corpus.beginning_labels,
			annotator_model = "seq",
			true_label_model = "HMM",
			discrete_feature_likelihoods = True,
			**kwargs
		)
		self.fitted = False

	def _init_bsc_model_inputs(self):
		# lookup object
		n_toks = np.asarray([doc.n_tokens for doc in self.corpus.docs]).sum().item()
		n_annotators = len(self.corpus.annotators)
		
		# initialize return objects
		features = list()
		doc_starts = np.zeros(n_toks, dtype=int)
		m = np.full((n_toks, n_annotators), -1, dtype=int)
		
		# iterate over docs and fill values in annotations matrix
		idx = 0
		for doc in self.corpus.docs:
			doc_starts[idx] = 1
			toks = doc.__getattribute__(self.tokens_field)
			features += toks
			n_ = doc.n_tokens
			for i, a in doc.annotations.items():
				m[idx:(idx+n_) , self.corpus.annotator_id2idx_[i]] = a
			idx += n_
		# return
		return(m, doc_starts, np.asarray(features))

	def _get_gold(self, label):
		n_labels, n_gold = 0, 0
		gold = []
		for doc in self.corpus.docs:
			tmp = np.full(doc.n_tokens, -1)
			if doc.n_labels > 0:
				n_labels += 1
				if label in doc.labels.keys():
					n_gold += 1
					gold.append(doc.labels[label])
				else:
					gold.append(tmp)
			else:
				gold.append(tmp)
		if n_labels > n_gold:
			print(f"Warning: {n_labels} labelled documents found, but only {n_gold} have entry for label '{label}'")
		return np.concatenate(gold)

	def reset_alpha_prior(self, new_alpha0):
		assert isinstance(new_alpha0, np.ndarray), "alpha0 must be a numpy ndarray"
		assert self.model.A.alpha_shape == new_alpha0.shape, f"shape of alpha0 mismatches. Expected shape: {self.model.A.alpha_shape}"
		assert (new_alpha0 > 0).all(), "All values in alpha0 must be positive."
		self.model.A.alpha0 = new_alpha0

	def reset_label_transitions_prior(self, new_beta0):
		assert isinstance(new_beta0, np.ndarray), "new_beta0 must be a numpy ndarray"
		assert self.model.LM.beta_shape == new_beta0.shape, f"shape of new_beta0 mismatches. Expected shape: {self.model.LM.beta_shape}"
		assert (new_beta0 > 0).all(), "All values in new_beta0 must be positive."
		self.model.LM.beta0 = new_beta0

	def fit_predict(self, refit = False, **kwargs):
		if self.fitted and not refit:
			raise Error("model already fitted to the data. Call method with `refit = True`to avoid this error message")
		
		self.runtime_ = timeit.default_timer()
		# fit and predict
		self.Et_, self.labels_, self.label_probs_ = \
			self.model.fit_predict(self.C, self.doc_start, self.features, **kwargs)
		
		self.runtime_ = timeit.default_timer() - self.runtime_
		self.fitted = True

		# get indexes where documents' sequences start
		sidxs = self.doc_start.nonzero()[0]
		# get indexes where documents' sequences end
		eidxs = sidxs[1:]
		eidxs = np.insert(eidxs, len(eidxs), len(self.doc_start))

		src = self.model_name
		# iterate over docs in corpus and add label
		for i, (s, e) in enumerate(zip(sidxs, eidxs)):
			if src not in self.corpus.docs[i].labels.keys():
				self.corpus.docs[i].add_labels(labels = self.labels_[s:e], source = src)

		# compute annotator parameters
		EPi = self.model.A._calc_EPi(self.model.A.alpha)
		self.annotator_params_ = {id: EPi[:,:,:,i] for id, i in self.corpus.annotator_id2idx_.items()}

	def write_labeled_to_json(self, fp, overwrite = False, encoding = "utf-8"):
		assert self.fitted, "Model has not yet been fitted. Call fit_predict() first!"
		if os.path.exists(fp) and not overwrite:
			raise FileExistsError(f"file '{fp}' already exists. Set `overwrite = True` to overwrite it.")
		out = dict()
		for doc in self.corpus.docs:
			out[doc.id] = dict(text = doc.text, tokens = doc.tokens, labels = doc.labels[self.model_name].tolist())
		with open(fp, "w", encoding=encoding) as f:
			json.dump(out, f)

	def get_label_distribution(self):
		if not hasattr(self, "labels_"):
			raise ValueError("attribute 'labels_' not set. call fit_predict() method first")
		labs = np.asarray([k for k, v in sorted(amodel.corpus.label_map.items(), key=lambda item: item[1])])
		l, n = np.unique(amodel.labels_, return_counts=True)
		return({l: n for l, n in zip(labs[l], n)})
	
	def extract_annotated_spans(self):
		"""
		Return: list of tuples of	strings. 
		A tuple's first element is the span in its original text;
		its second element is the span in its cleaned text version (taken from document's "tokens_cleaned_" field)
		"""
		if hasattr(self, "annotated_spans_"):
			return(self.annotated_spans_)
		# else compute
		spans = list()
		for doc in self.corpus.docs:
			ctoks = doc.tokens_cleaned_ if hasattr(doc, "tokens_cleaned_") else doc.tokens
			for l, t, c in zip(doc.labels[self.model_name], doc.tokens, ctoks):
				if l == self.corpus.outside_label:
					continue
				elif l in self.corpus.beginning_labels:
					spans.append([t, c])
				else:
					spans[-1][0] += (u"\x20" + t)
					spans[-1][1] += (u"\x20" + c)
		spans = [tuple(span) for span	in spans]
		# set attribute	
		self.annotated_spans_ = spans
		return(spans)

	def get_annotator_informativeness(self):
		annotator_ids = list(self.corpus.annotators.keys()) 
		out = pd.DataFrame(
			self.model.informativeness(), 
			index = annotator_ids, 
			columns = ["informativeness"]
		)
		return(out.sort_values("informativeness", ascending = False))

	def get_annotator_parameters(self):
		if not hasattr(self, "annotator_params_"):
			raise ValueError("attribute 'annotator_params_' not set. call fit_predict() method first")
		return(self.annotator_params_)

	def get_feature_class_probabilities(self):
		if not hasattr(self, "Et_"):
			raise ValueError("attribute 'Et_' not set. call fit_predict() method first")
		cols = [k for k, v in sorted(self.corpus.label_map.items(), key=lambda item: item[1])]
		df = pd.DataFrame(self.Et_, index = self.features, columns=cols)
		return(df)

	def compute_benchmark(self, benchmark = "MV"):
		if benchmark.lower() == "mv":
			self.mv_labels_, mv_plabels = MV(self.C, self.num_classes).vote(simple=False)
		else:
			raise NotImplementedError(f"benchmark = '{benchmark}' not implemented")

