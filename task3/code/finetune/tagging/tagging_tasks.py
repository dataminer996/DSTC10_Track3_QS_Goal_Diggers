# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sequence tagging tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import os
import re
import copy
import stanza
import ctypes
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf1 
import unicodedata

from finetune import feature_spec
from finetune import task
from finetune.tagging import tagging_metrics
from finetune.tagging import tagging_utils
from model import tokenization
from pretrain import pretrain_helpers
from util import utils


_file = 'get_sg_features.so'
_path = os.path.join(*(os.path.split(__file__)[:-1] + (_file,)))

#print(_path)
_mod = ctypes.cdll.LoadLibrary(_path)

# void avg(double *, int n)
# Define a special type for the 'double *' argument
class DoubleArrayType:
    def from_param(self, param):
        typename = type(param).__name__
        if hasattr(self, 'from_' + typename):
            return getattr(self, 'from_' + typename)(param)
        elif isinstance(param, ctypes.Array):
            return param
        else:
            raise TypeError("Can't convert %s" % typename)

    # Cast from array.array objects
    def from_array(self, param):
        if param.typecode != 'd':
            raise TypeError('must be an array of doubles')
        ptr, _ = param.buffer_info()
        return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))

    # Cast from lists/tuples
    def from_list(self, param):
        val = ((ctypes.c_double)*len(param))(*param)
        return val
    from_tuple = from_list
    # Cast from a numpy array
    def from_ndarray(self, param):
        return param.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Cast from lists/tuples
    def from_list(self, param):
        val = ((ctypes.c_double)*len(param))(*param)
        return val
    from_tuple = from_list
    # Cast from a numpy array
    def from_ndarray(self, param):
        return param.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

arr1 = DoubleArrayType()
arr2 = DoubleArrayType()
res_arr = DoubleArrayType()
_get_sg_matrix = _mod.get_sg_matrix
_get_sg_matrix.argtypes = (arr1, ctypes.c_int, arr2, ctypes.c_int, res_arr)
_get_sg_matrix.restype = ctypes.c_int


def sgnet_matrix(result):
    arr = np.eye(len(result),dtype=np.int32)
    for i in range(len(result)):
      if result[i][0].lower()=='root':
        root = result[i][2] 
    #print(root)
    for j in range(len(arr)):
        arr[j][root-1] = 1

    for i in range(1,len(result)):
        x = result[i][2]-1  #self
        y = result[i][1]-1  #parent
        # arr[x][y] = 1
        parent = y
        while parent != -1:   ## 最后肯定回到根节点
            for res in result:
                if res[2] == parent+1:
                    arr[x][parent] = 1
                    parent = res[1]-1
                    break
    return arr

def truncate_sequences(sequences, max_length):
    words_to_cut = sum(map(len, sequences)) - max_length
    if words_to_cut <= 0:
        return sequences

    while words_to_cut > 0:
        words_to_cut -= len(sequences[0])
        sequences = sequences[1:]

    return sequences


LABEL_ENCODING = "BIOES"


class TaggingExample(task.Example):
    """A single tagged input sequence."""
    def __init__(self, eid, task_name, words, pos_tags, slot_key, slot_value, label, pos_mapping, is_token_level, nlp):
        super(TaggingExample, self).__init__(task_name)
        self.eid = eid
        self.nlp = nlp
        self.words = words
        self.slot_key = slot_key
        self.slot_value = slot_value
        self.label = int(label)
        self.pos_words = words
        #print(pos_tags)
        self._pos_mapping = None
        self.pos_tags = [pos_mapping[pos] for pos in pos_tags]
        self.is_token_level = is_token_level


class TaggingTask(task.Task):
    """Defines a sequence tagging task (e.g., part-of-speech tagging)."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, config, name,
                 tokenizer, is_token_level):
        super(TaggingTask, self).__init__(config, name)
        self._tokenizer = tokenizer
        self._is_token_level = is_token_level
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')
        self._pos_mapping_path = os.path.join(
            self.config.traindata_dir,
            ("debug_" if self.config.debug else "") + self.name +
            "_pos_tags.pkl")
        # self._pos_mapping_path = os.path.join(
        #     self.config.data_dir,
        #     ("debug_" if self.config.debug else "") + self.name +
        #     "_pos_tags.pkl")
        self._pos_mapping = None

    def get_examples(self, split):
        sentences = self._get_labeled_sentences(split)
        # print(len(sentences))
        examples = []
        pos_mapping = self._get_label_mapping(split, sentences)
        for i, (eid, words, pos_tags, slot_key, slot_value, label) in enumerate(sentences):
            examples.append(
                TaggingExample(eid, self.name, words, pos_tags, slot_key, 
                  slot_value, label, pos_mapping, self._is_token_level, self.nlp))

        return examples
    
    def _get_label_mapping(self, provided_split=None, provided_sentences=None):
        if self._pos_mapping is not None:
          return self._pos_mapping
        
        if tf.io.gfile.exists(self._pos_mapping_path):
          self._pos_mapping = utils.load_pickle(self._pos_mapping_path)
          return self._pos_mapping

        utils.log("Writing label mapping for task", self.name)
        pos_counts = collections.Counter()
        for split in ["train", "dev", "test"]:
            if not tf.io.gfile.exists(
                    os.path.join(self.config.raw_data_dir(self.name),
                                 split + ".txt")):
                continue
            if split == provided_split:
                split_sentences = provided_sentences
            else:
                split_sentences = self._get_labeled_sentences(split)
            for eid, _, pos_tags, slot_key, _, _ in split_sentences:
                for pos in pos_tags:
                  if pos in pos_counts.keys():
                    pos_counts[pos] += 1
                  else:
                    pos_counts[pos] = 1

            pos_labels = sorted(pos_counts.keys())
            pos_mapping = {label: i for i, label in enumerate(pos_labels)}
        utils.write_pickle(pos_mapping, self._pos_mapping_path)
        self._pos_mapping = pos_mapping
        return self._pos_mapping


    def get_text_sg(self, bert_tokens, text):
      # print(text)
      text = " ".join(text.split())
      doc = self.nlp(text)
      # sg_relation = [(word.deprel, word.head, word.id) for sent in doc.sentences for word in sent.words]
      sg_tokens = [token.text for sent in doc.sentences for token in sent.tokens]
      # print(sg_relation) 
      # assert len(sg_relation) == len(sg_tokens)
      N = len(sg_tokens)
      # print(N)
      sg_tokens_matrix = np.zeros((N, N), dtype=np.int32)

      roots_relations = []
      flag = False
      
      root_num = 0
      for sent in doc.sentences:
        tmp = []
      # for each in sg_relation:
        # if each[0].lower() == 'root':
        #   if flag:
        #     roots_relations.append(tmp)
        #   flag = True
        #   tmp = []
        #   tmp.append(each)
        #   root_num += 1
        # else:
        for word in sent.words:
          tmp.append((word.deprel, word.head, word.id))
        root_num += 1
        roots_relations.append(tmp)
      assert root_num == len(roots_relations)
      #print(sg_tokens)
      #print(sg_relation)
      #print(roots_relations)

      token_num = 0
      offset = 0
      # print(roots_relations)
      for root in roots_relations:
        token_num += len(root)
        # print(len(root), token_num, offset)
        tmp_matrix = sgnet_matrix(root)
        # print(tmp_matrix.shape)
        # print(sg_tokens_matrix.shape)
        # print(sg_tokens_matrix[offset:token_num, offset:token_num].shape)
        sg_tokens_matrix[offset:token_num, offset:token_num] = tmp_matrix
        offset += len(root)

      char_to_tok_index = []  # 每个字符对应的token的index
      char_to_tok_char = []
      #print(sg_tokens)
      for i, token in enumerate(sg_tokens):
        str_list = [s for s in list(token) if s != ' ']   ## 每个token内的空格去掉 再建立索引。。。
        char_to_tok_index.extend([i] * len(str_list))
        char_to_tok_char.extend(str_list)

      assert len(char_to_tok_index) == len(char_to_tok_char)

      bert_tok_to_orig_index = []
      prev_len = 0
      #print(bert_tokens)
      # print(char_to_tok_index)
      # print(char_to_tok_char)
      for each in bert_tokens:
        if each == ['[CLS]'] or each == ['[SEP]']:
          bert_tok_to_orig_index.append(-100000)
          continue
        for token in each:
          #print(token) 
          new_token = token.replace("#", '').replace(' ', '') ## 替换token内的下划线和可能出现的空格 
          start_index = prev_len
          end_index = start_index + len(new_token) - 1
          if new_token == '[UNK]':
            map_index = char_to_tok_index[start_index]
            bert_tok_to_orig_index.append(map_index)
            prev_len += 1
            continue
          char_str = ''.join(char_to_tok_char[start_index:end_index+1]).lower()
          
          if new_token != char_str:
            print('-----------------------error')
            print(new_token)
            print(char_str)
            print(start_index)
            print(text)
            print(char_to_tok_index)
            print(char_to_tok_char)
            print(bert_tokens)
            pass
          if len(list(set(char_to_tok_index[start_index:end_index+1]))) != 1:  # 是不是横跨了
            pass

          map_index = char_to_tok_index[start_index]
          bert_tok_to_orig_index.append(map_index)
          prev_len += len(new_token)

      #print(bert_tokens)
      #print(sg_tokens)
      #print(char_to_tok_index)
      #print(char_to_tok_char)
      #print(bert_tok_to_orig_index)
      #print(sg_tokens_matrix)

      m = len(bert_tok_to_orig_index)     # 这里不是标准的max_seq_length长度
      for i in range(m, self.config.max_seq_length):
        bert_tok_to_orig_index.append(-100000)

      token_index = np.array(bert_tok_to_orig_index, dtype=np.int32)

      #print(dim)
      #print(sg_tokens_matrix.shape)                    
      feature_matrix = np.zeros((self.config.max_seq_length, self.config.max_seq_length), dtype=np.int32) # 256*256
      if len(sg_tokens) > self.config.max_seq_length:
          print("len(sg_tokens)", len(sg_tokens))
          return None
      ret = _get_sg_matrix(token_index, self.config.max_seq_length, sg_tokens_matrix, len(sg_tokens), feature_matrix)
      #print(feature_matrix)
      return feature_matrix


    def featurize(self, example, is_training, log=False):
        words_to_tokens_a = tokenize_and_align(self._tokenizer, example.words)
        words_to_tokens_b = tokenize_and_align(self._tokenizer, [example.slot_key, '[SEP]', example.slot_value])
        if words_to_tokens_b[0] == ['[CLS]']:
            words_to_tokens_b = words_to_tokens_b[1:]
        # print(words_to_tokens)
        # print(sum(map(len, words_to_tokens)))
        words_to_tokens_a = truncate_sequences(words_to_tokens_a, self.config.max_seq_length-1-len(words_to_tokens_b))
        # words_to_tokens = [["[CLS]"]] + words_to_tokens[0] + [["[SEP]"]]
        # print(words_to_tokens)
        # print(sum(map(len, words_to_tokens)))
        if words_to_tokens_a[0] != ["[CLS]"]:
          words_to_tokens_a = [["[CLS]"]] + words_to_tokens_a
        # print(words_to_tokens)
        # if words_to_tokens[0][-1] != ["[SEP]"]:
        #   words_to_tokens += [["[SEP]"]]
        # words_to_tokens = words_to_tokens[0]
        # print(words_to_tokens)
        input_ids = []
        tagged_positions = []
        segment_ids = []
        for word_tokens in words_to_tokens_a:
            if len(word_tokens) + len(
                    input_ids) + 2 > self.config.max_seq_length:
                input_ids.append(self._tokenizer.vocab["[SEP]"])
                segment_ids.append(0)
                break
            if "[CLS]" not in word_tokens and "[SEP]" not in word_tokens:
                tagged_positions.append(len(input_ids))
            for token in word_tokens:
                input_ids.append(self._tokenizer.vocab[token])
                segment_ids.append(0)

        for word_tokens in words_to_tokens_b:
            if len(word_tokens) + len(
                    input_ids) + 1 > self.config.max_seq_length:
                input_ids.append(self._tokenizer.vocab["[SEP]"])
                segment_ids.append(1)
                break
            if "[CLS]" not in word_tokens and "[SEP]" not in word_tokens:
                tagged_positions.append(len(input_ids))
            for token in word_tokens:
                input_ids.append(self._tokenizer.vocab[token])
                segment_ids.append(1)

        # pad = lambda x: x + [0] * (self.config.max_seq_length - len(x))
        # segment_ids = pad([1] * len(input_ids))
        # input_mask = pad([1] * len(input_ids))
        # input_ids = pad(input_ids)
        pad = lambda x: x + [0] * (self.config.max_seq_length - len(x))
        segment_ids = pad(segment_ids)
        input_mask = pad([1] * len(input_ids))
        input_ids = pad(input_ids)

        assert len(input_ids) == self.config.max_seq_length
        assert len(input_mask) == self.config.max_seq_length
        assert len(segment_ids) == self.config.max_seq_length

        # add pos tags
        pos_words_to_tokens = copy.deepcopy(words_to_tokens_a)
        #print(example.pos_words)
        #print(pos_words_to_tokens)
        pos_input_ids = []
        pos_tagged_positions = []
        for word_tokens in pos_words_to_tokens:
          if len(pos_words_to_tokens) + len(pos_input_ids) + 1 > self.config.max_seq_length:
            pos_input_ids.append(self._tokenizer.vocab["[SEP]"])
            break
          if "[CLS]" not in word_tokens and "[SEP]" not in word_tokens:
            pos_tagged_positions.append(len(pos_input_ids))
          for token in word_tokens:
            pos_input_ids.append(self._tokenizer.vocab[token])

        #pad = lambda x: x + [0] * (self.config.max_seq_length - len(x))
        pos_labels = pad(example.pos_tags[:self.config.max_seq_length])
        pos_labeled_positions = pad(pos_tagged_positions)
        pos_labels_mask = pad([1.0] * len(pos_tagged_positions))
        pos_segment_ids = pad([1] * len(pos_input_ids))
        pos_input_mask = pad([1] * len(pos_input_ids))
        pos_input_ids = pad(pos_input_ids)
        assert len(pos_input_ids) == self.config.max_seq_length
        assert len(pos_input_mask) == self.config.max_seq_length
        assert len(pos_segment_ids) == self.config.max_seq_length
        assert len(pos_labels) == self.config.max_seq_length
        assert len(pos_labels_mask) == self.config.max_seq_length

        # adding AAnet
        # words_to_tokens = tokenize_and_align(self._tokenizer, example.words)
        #if self.config.use_sgnet:
        #    text = ''
        #    for token in words_to_tokens_a:
        #        if token in [['[CLS]'], ['[SEP]']]:
        #            continue
        #        tmp = ''
        #        for t in token:
        #            if t.startswith('#'):
        #                tmp += t.replace("#", "")
        #            else:
        #                tmp += t
        #        en_pat = re.compile('[a-zA-Z0-9]+')
        #        if re.match(en_pat, tmp):
        #            text += ' ' + tmp
        #        else:
        #            text += tmp
        #    text = re.compile('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f]').sub(' ', text)
            # print(text)
            # print(words_to_tokens)
            # try:
        #    sg_matrix = self.get_text_sg(words_to_tokens_a, text.strip())
            # print('sg_matrix', sg_matrix)
        #    if sg_matrix is None:
        #        return None
        #else:
        sg_matrix = np.zeros((self.config.max_seq_length, self.config.max_seq_length), dtype=np.int32)

        # print('chunk_slot: ', slot)
        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "task_id": self.config.task_names.index(self.name),
            self.name + "_eid": example.eid,
            self.name + "_label": int(example.label),
            #self.name + "_slot_key": example.slot_key,
            #self.name + "_slot_value": example.slot_value,
            self.name + "_pos_tags": pos_labels,
            self.name + "_pos_mask": pos_labels_mask,
            self.name + "_pos_positions": pos_labeled_positions,
            self.name + '_sgnet_logits': sg_matrix.reshape(self.config.max_seq_length * self.config.max_seq_length, -1).flatten().tolist()
        }

    def _get_labeled_sentences(self, split):
        sentences = []
        count = 0
        print('reading: ', os.path.join(self.config.raw_data_dir(self.name),
                             split + ".txt"))
        with tf.io.gfile.GFile(
                os.path.join(self.config.raw_data_dir(self.name),
                             split + ".txt"), "r") as f:
        # print('reading: ', os.path.join(self.config.data_dir,
        #                      split + ".txt"))
        # with tf.io.gfile.GFile(
        #         os.path.join(self.config.data_dir,
        #                      split + ".txt"), "r") as f:
            for line in f:
                # print(count)
                # if count >= 500:
                #   break
                line = line.strip().split('\t')
                if line:
                    # print(line)
                #   if len(line) == 6:
                    chars, pos_tags, _, slot_key, slot_value, label, eid = line
                    chars = chars.split('#')
                    pos_tags = pos_tags.split('#')
                    sentences.append((int(eid), chars, pos_tags, slot_key, slot_value, int(label)))
                    count += 1
            
        return sentences

    def get_scorer(self):
        return tagging_metrics.AccuracyScorer() if self._is_token_level else \
            tagging_metrics.EntityLevelF1Scorer(self._get_label_mapping())

    def get_feature_specs(self):
        return [
            feature_spec.FeatureSpec(self.name + "_eid", []),
            #feature_spec.FeatureSpec(self.name + "_slot_key", []),
            #feature_spec.FeatureSpec(self.name + "_slot_value", []),
            feature_spec.FeatureSpec(self.name + "_pos_tags",
                                     [self.config.max_seq_length]),
            feature_spec.FeatureSpec(self.name + "_pos_mask",
                                     [self.config.max_seq_length],
                                     is_int_feature=False),
            feature_spec.FeatureSpec(self.name + "_pos_positions",
                                     [self.config.max_seq_length]),
            feature_spec.FeatureSpec(
                self.name + "_sgnet_logits",
                [self.config.max_seq_length * self.config.max_seq_length]),
            feature_spec.FeatureSpec(self.name + "_label", [])
        ]

    def create_model_crf(self, bert_model, label, num_labels,
             mode, input_ids, cx_labels, cx_mask, cx_num_labels, sg_labels):
        output_layer = bert_model.get_sequence_output()
        # pool_output_layer = bert_model.get_pooled_output()
        # hidden_size = output_layer.shape[-1].value
        hidden_size = output_layer.shape[-1]
        # print(output_layer)
        print("===================mode", mode)
        print(tf.estimator.ModeKeys.PREDICT)
        output_label_weight = tf.get_variable(
            "output_label_weight", [1, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.05))
        output_label_bias = tf.get_variable(
            "output_label_bias", [1], initializer=tf.zeros_initializer())

        with tf.variable_scope("cixing_loss"):

            # if mode == tf.estimator.ModeKeys.TRAIN:
            #    output_layer_drop = tf.nn.dropout(
            #        output_layer, keep_prob=self.config.entity_dropout)

            cx_logits = tf.layers.dense(output_layer,
                                        cx_num_labels,
                                        name='cx_dense')
            print(cx_logits)

            if mode != tf.estimator.ModeKeys.PREDICT:
                cx_loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(cx_labels, cx_num_labels),
                    logits=cx_logits)
                print(cx_logits)
                print(cx_loss)
                cx_loss *= cx_mask
                print(cx_loss)
                cx_loss = tf.reduce_sum(cx_loss, axis=-1)
                print(cx_loss)
                cx_loss = tf.reduce_mean(cx_loss, axis=-1)
                print(cx_loss)

                # cx_log_probs = tf.nn.log_softmax(cx_logits, axis=-1)
                cx_one_hot_labels = tf.one_hot(cx_labels,
                                               depth=cx_num_labels,
                                               dtype=tf.float32)
                print(cx_logits)
                print(tf.argmax(cx_logits, -1))
                print(cx_labels)
                cx_correct_prediction = tf.equal(tf.cast(cx_labels, tf.int64),
                                                 tf.argmax(cx_logits, -1))
                cx_correct_prediction = tf.cast(cx_correct_prediction,
                                                tf.float32)
                cx_acc = tf.reduce_mean(cx_correct_prediction)
                cx_acc = tf.reduce_mean(cx_acc)
                print(cx_one_hot_labels)
                print("is traingdebug ===========")
            else:
                cx_loss = tf.constant([0.0], dtype=tf.float32)
                cx_acc = tf.constant([0.0], dtype=tf.float32)
            # cx_probabilities = tf.nn.softmax(cx_logits, axis=-1)
            # cx_predict = tf.argmax(cx_probabilities, axis=-1)

        with tf.variable_scope("sg_loss"):

            # if mode == tf.estimator.ModeKeys.TRAIN:
            # output_layer_drop = tf.nn.dropout(
            #   output_layer, keep_prob=self.config.entity_dropout)
            sg_num_labels = self.config.max_seq_length
            sg_logits = tf.layers.dense(output_layer,
                                        sg_num_labels,
                                        name='sg_dense')
            sg_logits = tf.reshape(sg_logits,
                                   [-1, sg_num_labels * sg_num_labels])
            print(sg_logits)

            if mode != tf.estimator.ModeKeys.PREDICT:
                sg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(sg_labels, tf.float32), logits=sg_logits)
                print(sg_logits)
                print(sg_loss)
                # sg_loss *= sg_mask
                # print(sg_loss)
                sg_loss = tf.reduce_sum(sg_loss, axis=-1)
                print(sg_loss)
                sg_loss = tf.reduce_mean(sg_loss, axis=-1)
                print(sg_loss)

                # sg_log_probs = tf.nn.log_softmax(sg_logits, axis=-1)
                # sg_one_hot_labels = tf.one_hot(sg_labels, depth=sg_num_labels, dtype=tf.float32)
                print(sg_logits)
                print(tf.argmax(sg_logits, -1))
                print(sg_labels)
                sg_predict = tf.cast(tf.round((tf.sign(sg_logits) + 1) / 2),
                                     tf.int32)
                sg_correct_prediction = tf.equal(tf.cast(sg_labels, tf.int32),
                                                 sg_predict)
                sg_correct_prediction = tf.cast(sg_correct_prediction,
                                                tf.float32)
                sg_acc = tf.reduce_mean(sg_correct_prediction)
                sg_acc = tf.reduce_mean(sg_acc)
                # print(sg_one_hot_labels)
                print("is traingdebug ===========")
            else:
                sg_loss = tf.constant([0.0], dtype=tf.float32)
                sg_acc = tf.constant([0.0], dtype=tf.float32)
            # sg_probabilities = tf.math.sigmoid(sg_logits)
            sg_predict = tf.cast(tf.round((tf.sign(sg_logits) + 1) / 2),
                                 tf.int32)

        with tf.variable_scope("label_loss"):
            label_output_layer = tf.reshape(output_layer, [-1, hidden_size])
            label_logits = tf.matmul(label_output_layer,
                                      output_label_weight,
                                      transpose_b=True)
            if mode == tf.estimator.ModeKeys.TRAIN:
                label_logits = tf.nn.dropout(
                    label_logits, keep_prob=self.config.label_dropout)

            label_logits = tf.nn.bias_add(label_logits, output_label_bias)
            # intent_logits = tf.layers.dense(output_layer, 1)
            label_logits = tf.reshape(label_logits,
                                      [-1, self.config.max_seq_length])
            label_logits = tf.layers.dense(
                label_logits,
                num_labels,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.02),
                name="label_dense",
                use_bias=True)
            if mode != tf.estimator.ModeKeys.PREDICT:
                label_correct_prediction = tf.equal(
                    tf.cast(label, tf.int64),
                    tf.argmax(label_logits, -1))
                label_correct_prediction = tf.cast(label_correct_prediction,
                                                    tf.float32)
                label_acc = tf.reduce_mean(label_correct_prediction)

                label_one_hot_labels = tf.one_hot(label,
                                                   depth=num_labels,
                                                   dtype=tf.float32)
                label_loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=label_one_hot_labels, logits=label_logits)

                print(label_loss)
                label_loss = tf.reduce_mean(label_loss, name='label_loss')
                print("is traingdebug ===========")
            else:
                label_loss = tf.constant([0.0], dtype=tf.float32)
                label_acc = tf.constant([0.0], dtype=tf.float32)

            print(label_loss)

            label_probabilities = tf.nn.softmax(label_logits, axis=-1)
            label_predict = tf.argmax(label_probabilities, axis=-1)

            total_loss = self.config.label_weight * label_loss + self.config.cx_weight * cx_loss + self.config.sg_weight * sg_loss
            print(total_loss)

            return (total_loss, label_loss, label_probabilities, label_predict, label_acc, cx_loss, cx_acc, sg_loss,
                    sg_acc)

    def get_prediction_module(self, bert_model, features, is_training,
                              percent_done, mode):
        # n_classes = len(self._get_label_mapping())
        reprs = bert_model.get_sequence_output()
        # reprs = bert_model.get_pooled_output()

        # reprs = pretrain_helpers.gather_positions(
        #     reprs, tf.cast(features[self.name + "_entity_positions"],
        #                    tf.int32))

        num_labels = self.config.num_labels
        cixing_num_labels = self.config.cixing_num_labels


        (losses, label_loss, label_probabilities, label_predict, label_acc, cx_loss, cx_acc, sg_loss,
            sg_acc) = self.create_model_crf(
             bert_model, features[self.name + "_label"], num_labels, mode,
             features["input_ids"], features[self.name + "_pos_tags"],
             features[self.name + "_pos_mask"], cixing_num_labels,
             features[self.name + "_sgnet_logits"])

        if mode != tf.estimator.ModeKeys.TRAIN:
            return losses, dict(
                label_predict=label_predict,
                label_probabilities=label_probabilities,
                labels=features[self.name + "_label"],
                eid=features[self.name + "_eid"],
            )
        else:
            return losses, dict(
                loss=losses,
                label_loss=label_loss,
                cx_loss=cx_loss,
                cx_acc=cx_acc,
                sg_loss=sg_loss,
                sg_acc=sg_acc,
                label_predict=label_predict,
                label_probabilities=label_probabilities,
                label_accuracy=label_acc,
                labels=features[self.name + "_label"],
                eid=features[self.name + "_eid"],
            )


def tokenize_and_align(tokenizer, words, cased=False):
    """Splits up words into subword-level tokens."""
    words = ["[CLS]"] + list(words) + ["[SEP]"]
    basic_tokenizer = tokenizer.basic_tokenizer
    tokenized_words = []
    for word in words:
        word = tokenization.convert_to_unicode(word)
        word = basic_tokenizer._clean_text(word)
        if word == "[CLS]" or word == "[SEP]":
            word_toks = [word]
        else:
            if not cased:
                word = word.lower()
                word = basic_tokenizer._run_strip_accents(word)
            word_toks = basic_tokenizer._run_split_on_punc(word)
        tokenized_word = []
        for word_tok in word_toks:
            tokenized_word += tokenizer.wordpiece_tokenizer.tokenize(word_tok)
        tokenized_words.append(tokenized_word)
    assert len(tokenized_words) == len(words)
    return tokenized_words


class Chunking(TaggingTask):
    """Text chunking."""
    def __init__(self, config, tokenizer):
        super(Chunking, self).__init__(config, "chunk", tokenizer, False)
