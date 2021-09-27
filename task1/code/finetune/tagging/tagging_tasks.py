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
import tensorflow.keras.backend as K

from finetune import feature_spec
from finetune import task
from finetune.tagging import tagging_metrics
from finetune.tagging import tagging_utils
from model import tokenization
from pretrain import pretrain_helpers
from util import utils
import torch
torch.set_printoptions(profile='full')
np.set_printoptions(threshold=np.inf)

#stanza.download('en')
print(os.path.abspath(__file__))
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
    def __init__(self, eid, task_name, words, pos_tags, action, disambiguate_label, slot, from_system, objects_num, is_token_level,
                 action_mapping, slot_mapping, pos_mapping, nlp):
        super(TaggingExample, self).__init__(task_name)
        self.eid = eid
        self.nlp = nlp
        self.words = words
        if action != '':
            self.action = action_mapping[action]
        else:
            self.action = -1
            
        self.action_words = action
        if disambiguate_label != '':
            self.disambiguate = int(disambiguate_label)
        else:
            self.disambiguate = -1
        self.from_system = from_system
        self.objects_num = objects_num
        self.pos_words = words
        #print(pos_tags)
        self.pos_tags = [pos_mapping[pos] for pos in pos_tags]
        self.slot = [0 for l in range(len(slot_mapping))]
        # print('------------------------------')
        # print(slot_mapping)
        # print(slot)
        for s in slot:
          self.slot[slot_mapping[s]] = 1
        # print(self.slot)
        self.is_token_level = is_token_level


class TaggingTask(task.Task):
    """Defines a sequence tagging task (e.g., part-of-speech tagging)."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, config, name,
                 tokenizer, is_token_level):
        super(TaggingTask, self).__init__(config, name)
        self._tokenizer = tokenizer
        self._action_mapping_path = os.path.join(
            self.config.traindata_dir,
            ("debug_" if self.config.debug else "") + self.name +
            "_action.pkl")
        self._is_token_level = is_token_level
        self._pos_mapping_path = os.path.join(
            self.config.traindata_dir,
            ("debug_" if self.config.debug else "") + self.name +
            "_pos_tags.pkl")
        self._slot_mapping_path = os.path.join(
            self.config.traindata_dir,
            ("debug_" if self.config.debug else "") + self.name +
            "_slot_types.pkl")
        print("=====slot mapping path: ", self._slot_mapping_path)
        self._action_mapping = None
        self._slot_mapping = None
        self._pos_mapping = None
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')

    def get_examples(self, split):
        sentences = self._get_labeled_sentences(split)
        # print(len(sentences))
        examples = []
        action_mapping, slot_mapping, pos_mapping = self._get_label_mapping(split, sentences)
        for i, (eid, words, pos_tags, slot_key, disambiguate_label, action, from_system, objects_num) in enumerate(sentences):
            examples.append(
                TaggingExample(eid, self.name, words, pos_tags, action, disambiguate_label, 
                  slot_key, from_system, objects_num, self._is_token_level, action_mapping, slot_mapping, pos_mapping, self.nlp))

        return examples

    def _get_label_mapping(self, provided_split=None, provided_sentences=None):
        if self._action_mapping is not None and self._slot_mapping is not None and \
            self._pos_mapping is not None:
          return self._action_mapping, self._slot_mapping, self._pos_mapping
        
        if tf.io.gfile.exists(self._action_mapping_path) and tf.io.gfile.exists(self._slot_mapping_path) \
            and tf.io.gfile.exists(self._pos_mapping_path):
          self._action_mapping = utils.load_pickle(self._action_mapping_path)
          self._slot_mapping = utils.load_pickle(self._slot_mapping_path)
          self._pos_mapping = utils.load_pickle(self._pos_mapping_path)
          return self._action_mapping, self._slot_mapping, self._pos_mapping

        utils.log("Writing label mapping for task", self.name)
        pos_counts = collections.Counter()
        action_counts = collections.Counter()
        slot_counts = collections.Counter()
        for split in ["train", "dev", "test"]:
            if not tf.io.gfile.exists(
                    os.path.join(self.config.raw_data_dir(self.name),
                                 split + ".txt")):
                continue
            if split == provided_split:
                split_sentences = provided_sentences
            else:
                split_sentences = self._get_labeled_sentences(split)
            for _, pos_tags, _,action,slot_key, _, _, _ in split_sentences:
                for pos in pos_tags:
                  if pos in pos_counts.keys():
                    pos_counts[pos] += 1
                  else:
                    pos_counts[pos] = 1
                
                for slot in slot_key:
                  if slot in slot_counts.keys():
                    slot_counts[slot] += 1
                  else:
                    slot_counts[slot] = 1

                if action in action_counts.keys():
                    action_counts[action] += 1
                else:
                    action_counts[action] = 1

            pos_labels = sorted(pos_counts.keys())
            pos_mapping = {label: i for i, label in enumerate(pos_labels)}
            action_labels = sorted(action_counts.keys())
            action_mapping = {label: i for i, label in enumerate(action_labels)}
            slot_labels = sorted(slot_counts.keys())
            slot_mapping = {label: i for i, label in enumerate(slot_labels)}
        utils.write_pickle(pos_mapping, self._pos_mapping_path)
        utils.write_pickle(action_mapping, self._action_mapping_path)
        utils.write_pickle(slot_mapping, self._slot_mapping_path)
        self._action_mapping = action_mapping
        self._slot_mapping = slot_mapping
        self._pos_mapping = pos_mapping
        return self._action_mapping, self._slot_mapping, self._pos_mapping

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
        # print(root)
        # print(tmp_matrix)
        # print(tmp_matrix.shape)
        # print(sg_tokens_matrix.shape)
        # print(sg_tokens_matrix[offset:token_num, offset:token_num].shape)
        sg_tokens_matrix[offset:token_num, offset:token_num] = tmp_matrix
        offset += len(root)
    #   print(sg_tokens_matrix)
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
    #   print(bert_tokens)
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
    #   print('-----------------------------')
    #   print(bert_tokens)
    #   print(sg_tokens)
    #   print(char_to_tok_index)
    #   print(char_to_tok_char)
    #   print(bert_tok_to_orig_index)
    #   print(sg_tokens_matrix)

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
    #   print(feature_matrix)
      return feature_matrix, sg_tokens, roots_relations

    def get_text_aa(self, bert_tokens, text, sg_tokens, roots_relations):
        N = len(sg_tokens)
        # print(N)
        aa_tokens_matrix = np.zeros((N, N), dtype=np.int32)
        token_num = 0
        offset = 0
        # print(roots_relations)
        for root in roots_relations:
            token_num += len(root)
            sub_aanet = np.ones((len(root), len(root)), dtype=np.int32)
            # print("sub_aanet: ", sub_aanet.shape)
            aa_tokens_matrix[offset:token_num, offset:token_num] = sub_aanet
            offset += len(root)
        # print(aa_tokens_matrix.shape)
        char_to_tok_index = []  # 每个字符对应的token的index
        char_to_tok_char = []
        # print(sg_tokens)
        for i, token in enumerate(sg_tokens):
            str_list = [s for s in list(token) if s != ' ']  ## 每个token内的空格去掉 再建立索引。。。
            char_to_tok_index.extend([i] * len(str_list))
            char_to_tok_char.extend(str_list)

        assert len(char_to_tok_index) == len(char_to_tok_char)

        bert_tok_to_orig_index = []
        prev_len = 0
        #   print(bert_tokens)
        for each in bert_tokens:
            if each == ['[CLS]'] or each == ['[SEP]']:
                bert_tok_to_orig_index.append(-100000)
                continue
            for token in each:
                # print(token)
                new_token = token.replace("#", '').replace(' ', '')  ## 替换token内的下划线和可能出现的空格
                start_index = prev_len
                end_index = start_index + len(new_token) - 1
                if new_token == '[UNK]':
                    map_index = char_to_tok_index[start_index]
                    bert_tok_to_orig_index.append(map_index)
                    prev_len += 1
                    continue
                char_str = ''.join(char_to_tok_char[start_index:end_index + 1]).lower()

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
                if len(list(set(char_to_tok_index[start_index:end_index + 1]))) != 1:  # 是不是横跨了
                    pass

                map_index = char_to_tok_index[start_index]
                bert_tok_to_orig_index.append(map_index)
                prev_len += len(new_token)
        #   print('-----------------------------')
        #   print(bert_tokens)
        #   print(sg_tokens)
        #   print(char_to_tok_index)
        #   print(char_to_tok_char)
        #   print(bert_tok_to_orig_index)
        #   print(sg_tokens_matrix)

        m = len(bert_tok_to_orig_index)  # 这里不是标准的max_seq_length长度
        for i in range(m, self.config.max_seq_length):
            bert_tok_to_orig_index.append(-100000)

        token_index = np.array(bert_tok_to_orig_index, dtype=np.int32)

        feature_matrix_1 = np.zeros((self.config.max_seq_length, self.config.max_seq_length), dtype=np.int32)
        if len(sg_tokens) > self.config.max_seq_length:
            print("len(sg_tokens)", len(sg_tokens))
            return None, None
        ret1 = _get_sg_matrix(token_index, self.config.max_seq_length, aa_tokens_matrix, len(sg_tokens),
                              feature_matrix_1)
        #   print(feature_matrix_1.shape)
        #   print(feature_matrix)
        return feature_matrix_1


    def featurize(self, example, is_training, log=False):
        words_to_tokens = tokenize_and_align(self._tokenizer, example.words)
        # print(words_to_tokens)
        # print(sum(map(len, words_to_tokens)))
        words_to_tokens = truncate_sequences(words_to_tokens, self.config.max_seq_length-1)
        # words_to_tokens = [["[CLS]"]] + words_to_tokens[0] + [["[SEP]"]]
        # print(words_to_tokens)
        # print(sum(map(len, words_to_tokens)))
        if words_to_tokens[0] != ["[CLS]"]:
          words_to_tokens = [["[CLS]"]] + words_to_tokens
        
        words_to_tokens_b = tokenize_and_align(self._tokenizer, [example.action_words])
        if words_to_tokens_b[0] == ['[CLS]']:
            words_to_tokens_b = words_to_tokens_b[1:]
        # print(words_to_tokens)
        # if words_to_tokens[0][-1] != ["[SEP]"]:
        #   words_to_tokens += [["[SEP]"]]
        # words_to_tokens = words_to_tokens[0]
        # print(words_to_tokens)
        input_ids = []
        tagged_positions = []
        segment_ids = []
        for word_tokens in words_to_tokens:
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
        
        # for word_tokens in words_to_tokens_b:
        #     if len(word_tokens) + len(
        #             input_ids) + 1 > self.config.max_seq_length:
        #         input_ids.append(self._tokenizer.vocab["[SEP]"])
        #         segment_ids.append(1)
        #         break
        #     if "[CLS]" not in word_tokens and "[SEP]" not in word_tokens:
        #         tagged_positions.append(len(input_ids))
        #     for token in word_tokens:
        #         input_ids.append(self._tokenizer.vocab[token])
        #         segment_ids.append(1)

        pad = lambda x: x + [0] * (self.config.max_seq_length - len(x))
        segment_ids = pad([1] * len(input_ids))
        # segment_ids = pad(segment_ids)
        input_mask = pad([1] * len(input_ids))
        input_ids = pad(input_ids)
        # slot = pad(example.slot[:self.config.max_seq_length])
        slot = example.slot
        # print(slot)
        assert len(slot) == self.config.slot_num_labels
        assert len(input_ids) == self.config.max_seq_length
        assert len(input_mask) == self.config.max_seq_length
        assert len(segment_ids) == self.config.max_seq_length

        # add pos tags
        pos_words_to_tokens = copy.deepcopy(words_to_tokens)
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
        text = ''
        for token in words_to_tokens:
            if token in [['[CLS]'], ['[SEP]']]:
                continue
            tmp = ''
            for t in token:
                if t.startswith(''):
                    tmp += t.replace("#", "")
                else:
                    tmp += token
            en_pat = re.compile('[a-zA-Z0-9]+')
            if re.match(en_pat, tmp):
                text += ' ' + tmp
            else:
                text += tmp
        # print(text)
        # print(words_to_tokens)
        # try:
        if self.config.use_sgnet:
            sg_matrix, sg_tokens, roots_relations = self.get_text_sg(words_to_tokens, text.strip())
            aa_matrix = self.get_text_aa(words_to_tokens, text.strip(), sg_tokens, roots_relations)
            # print('sg_matrix', sg_matrix)
            if sg_matrix is None:
                return None
        else:
            sg_matrix = np.zeros((self.config.max_seq_length, self.config.max_seq_length), dtype=np.int32)
            aa_matrix = np.zeros((self.config.max_seq_length, self.config.max_seq_length), dtype=np.int32)
        # print('--------------------------------')
        # print(example.words)
        # print(words_to_tokens)
        # print("input_ids", input_ids)
        # print("input_mask", input_mask)
        # print("segment_ids", segment_ids)
        # print("task_id", self.config.task_names.index(self.name))
        # print("_eid", example.eid)
        # print("_action", int(example.action))
        # print("_disambiguate", int(example.disambiguate))
        # print("_slot", slot)
        # print("_pos_tags", pos_labels)
        # print("_pos_mask", pos_labels_mask)
        # print("_pos_positions", pos_labeled_positions)
        # print('_sgnet_logits', sg_matrix.reshape(self.config.max_seq_length * self.config.max_seq_length, -1).flatten().tolist())
        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "task_id": self.config.task_names.index(self.name),
            self.name + "_eid": example.eid,
            self.name + "_action": int(example.action),
            self.name + "_disambiguate": int(example.disambiguate),
            self.name + "_from_system": int(example.from_system),
            self.name + "_objects_num": int(example.objects_num),
            self.name + "_slot": slot,
            self.name + "_pos_tags": pos_labels,
            self.name + "_pos_mask": pos_labels_mask,
            self.name + "_pos_positions": pos_labeled_positions,
            self.name + '_sgnet_logits': sg_matrix.reshape(self.config.max_seq_length * self.config.max_seq_length, -1).flatten().tolist(),
            self.name + '_aanet_logits': aa_matrix.reshape(self.config.max_seq_length * self.config.max_seq_length, -1).flatten().tolist()
        }

    def _get_labeled_sentences(self, split):
        sentences = []
        count = 0
        print('reading: ', os.path.join(self.config.raw_data_dir(self.name),
                             split + ".txt"))
        with tf.io.gfile.GFile(
                os.path.join(self.config.raw_data_dir(self.name),
                             split + ".txt"), "r") as f:
            for line in f:
                # if count >= 5:
                #   break
                line = line.strip().split('\t')
                if line:
                  if len(line) == 8:
                    chars, pos_tags, disambiguate_label, action_label, slot_key, from_system, objects_num, eid = line
                    chars = chars.split('#')
                    pos_tags = pos_tags.split('#')
                    slot_key = [s for s in slot_key.split('#') if len(s) > 0]
                    sentences.append((int(eid), chars, pos_tags, slot_key, disambiguate_label, action_label, int(from_system), int(objects_num)))
                    count += 1
            
        return sentences

    def get_scorer(self):
        return tagging_metrics.AccuracyScorer() if self._is_token_level else \
            tagging_metrics.EntityLevelF1Scorer(self._get_label_mapping())

    def get_feature_specs(self):
        return [
            feature_spec.FeatureSpec(self.name + "_eid", []),
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
            feature_spec.FeatureSpec(
                self.name + "_aanet_logits",
                [self.config.max_seq_length * self.config.max_seq_length]),
            feature_spec.FeatureSpec(self.name + "_action", []),
            feature_spec.FeatureSpec(self.name + "_disambiguate", []),
            feature_spec.FeatureSpec(self.name + "_from_system", []),
            feature_spec.FeatureSpec(self.name + "_objects_num", []),
            feature_spec.FeatureSpec(self.name + "_slot", [self.config.slot_num_labels]),
            feature_spec.FeatureSpec("eid", []),
        ]
    
    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        """多标签分类的交叉熵
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
             1表示对应的类为目标类，0表示对应的类为非目标类。
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = K.zeros_like(y_pred[..., :1])
        y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
        y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
        neg_loss = K.logsumexp(y_pred_neg, axis=-1)
        pos_loss = K.logsumexp(y_pred_pos, axis=-1)
        return neg_loss + pos_loss
        
    def test_softmax_focal_ce_3(self, n_classes, gamma, alpha, logits, label):
        epsilon = 1.e-8
        # y_true and y_pred
        # y_true = tf.one_hot(label, n_classes)
        y_true = label
        probs = tf.nn.softmax(logits)
        y_pred = tf.clip_by_value(probs, epsilon, 1. - epsilon)
    
        # weight term and alpha term【因为y_true是只有1个元素为1其他元素为0的one-hot向量，所以对于每个样本，只有y_true位置为1的对应类别才有weight，其他都是0】这也是为什么网上有的版本会用到tf.gather函数，这个函数的作用就是只把有用的这个数取出来，可以省略一些0相关的运算。
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., y_pred), gamma))
        if alpha != 0.0:  # 我这实现中的alpha只是起到了调节loss倍数的作用（调节倍数对训练没影响，因为loss的梯度才是影响训练的关键），要想起到调节类别不均衡的作用，要替换成数组，数组长度和类别总数相同，每个元素表示对应类别的权重。另外[这篇](https://blog.csdn.net/Umi_you/article/details/80982190)博客也提到了，alpha在多分类Focal loss中没作用，也就是只能调节整体loss倍数，不过如果换成数组形式的话，其实是可以达到缓解类别不均衡问题的目的。
            alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        else:
            alpha_t = tf.ones_like(y_true)
    
        # origin x ent，这里计算原始的交叉熵损失
        xent = tf.multiply(y_true, -tf.log(y_pred))
    
        # focal x ent，对交叉熵损失进行调节，“-”号放在上一行代码了，所以这里不需要再写“-”了。
        focal_xent = tf.multiply(alpha_t, tf.multiply(weight, xent))
    
        # in this situation, reduce_max is equal to reduce_sum，因为经过y_true选择后，每个样本只保留了true label对应的交叉熵损失，所以使用max和使用sum是同等作用的。
        reduced_fl = tf.reduce_max(focal_xent, axis=1)
        return tf.reduce_mean(reduced_fl)
        
    def test_softmax_cross_entropy_with_logits(self, logits, y_true):
        epsilon = 1.e-8
        softmax_prob = tf.nn.softmax(logits) * tf.transpose([3.0, 2.0, 1.0])
        y_pred = tf.clip_by_value(softmax_prob, epsilon, 1. - epsilon)
        # 得到交叉熵，其中的“-”符号可以放在好几个地方，都是等效的，最后取mean是为了兼容batch训练的情况。
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true*tf.log(y_pred)))
        return cross_entropy
        
    def CE_with_prior(self, labels, logits, base_probs, num_labels, tau=1.0):
        '''
        param: one_hot_label
        param: logits
        param: prior: real data distribution obtained by statistics
        param: tau: regulator, default is 1
        return: loss
        '''   
        from_system_one_hot_labels = tf.one_hot(labels,
                                                  depth=num_labels,
                                                  dtype=tf.float32)
        base_probs = base_probs*from_system_one_hot_labels
        logits = logits + tf.math.log(tf.cast(base_probs**tau + 1e-12, dtype=tf.float32))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # prior = base_probs*labels
        # log_prior = K.constant(np.log(prior + 1e-8))
    
        # # align dim 
        # for _ in range(K.ndim(logits) - 1):     
        #     log_prior = K.expand_dims(log_prior, 0)
    
        # logits = logits + tau * log_prior
        # loss = K.categorical_crossentropy(labels, logits, from_logits=True)
    
        return loss


    def create_model(self, bert_model, action_label, action_num_labels,
            disambiguate_label, disambiguate_num_labels, slot_label, slot_num_labels,
             mode, input_ids, cx_labels, cx_mask, cx_num_labels, sg_labels, from_system_label, from_system_num_labels, objects_num_label, objects_num_num_labels, aa_labels, aa_num_labels):
        output_layer = bert_model.get_sequence_output()
        # pool_output_layer = bert_model.get_pooled_output()
        # hidden_size = output_layer.shape[-1].value
        hidden_size = output_layer.shape[-1]
        # print(output_layer)
        print("===================mode", mode)
        print(tf.estimator.ModeKeys.PREDICT)
        print(action_label)
        print(slot_label)
        # output_weight = tf.get_variable(
        #    "output_weights", [entity_num_labels, hidden_size],
        #    initializer=tf.truncated_normal_initializer(stddev=0.05))
        # output_bias = tf.get_variable("output_bias", [entity_num_labels],
        #                              initializer=tf.zeros_initializer())
        output_action_weight = tf.get_variable(
            "output_action_weight", [1, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.05))
        output_action_bias = tf.get_variable(
            "output_action_bias", [1], initializer=tf.zeros_initializer())

        output_disambiguate_weight = tf.get_variable(
            "output_disambiguate_weight", [1, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.05))
        output_disambiguate_bias = tf.get_variable(
            "output_disambiguate_bias", [1], initializer=tf.zeros_initializer())
        
        output_from_system_weight = tf.get_variable(
            "output_system_weight", [1, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.05))
        output_from_system_bias = tf.get_variable(
            "output_system_bias", [1], initializer=tf.zeros_initializer())
            
        output_objects_num_weight = tf.get_variable(
            "output_objects_num_weight", [1, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.05))
        output_objects_num_bias = tf.get_variable(
            "output_objects_num_bias", [1], initializer=tf.zeros_initializer())
            
        output_slot_weight = tf.get_variable(
            "output_slot_weight", [1, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.05))
        output_slot_bias = tf.get_variable(
            "output_slot_bias", [1], initializer=tf.zeros_initializer())

        with tf.variable_scope("slot_loss"):
            slot_label=tf.cast(slot_label,tf.float32)
            print('slot_label', slot_label, isinstance(slot_label,list))
            
            slot_output_layer = tf.reshape(output_layer, [-1, hidden_size])
            # print('slot output_layer', slot_output_layer)
            slot_logits = tf.matmul(slot_output_layer,
                                      output_slot_weight,
                                      transpose_b=True)
            if mode == tf.estimator.ModeKeys.TRAIN:
                slot_logits = tf.nn.dropout(slot_logits, keep_prob=self.config.slot_dropout)
            slot_logits = tf.nn.bias_add(slot_logits, output_slot_bias)
                
            # slot_logits = tf.layers.dense(slot_output_layer, slot_num_labels)
            
            # print("output_slot_weight", output_slot_weight)
            

            
            # print('slot logits', slot_logits)
            slot_logits = tf.reshape(slot_logits,
                                      [-1, self.config.max_seq_length])
            slot_logits = tf.layers.dense(
                slot_logits,
                slot_num_labels,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.02),
                name="slot_dense",
                use_bias=True)
           
            # slot_logits = tf.layers.dense(output_layer,
            #                                 slot_num_labels,
            #                                 name='slot_dense')
            print('slot logits', slot_logits)
            
            if mode != tf.estimator.ModeKeys.PREDICT:
                # log_likelihood, _ = tf1.contrib.crf.crf_log_likelihood(
                #     slot_logits, tf.cast(slot_one_hot_labels, tf.int32), seq_batch, crf_params)
                # slot_loss = tf.reduce_mean(-log_likelihood)
                # slot_loss = tf.nn.weighted_cross_entropy_with_logits(
                #     labels=slot_label, logits=slot_logits, pos_weight = 2.0)
                # slot_loss = tf.reduce_mean(slot_loss, name='slot_loss')
                # slot_loss = self.multilabel_categorical_crossentropy(slot_label, slot_logits)
                # pt = tf.nn.sigmoid(slot_logits)
                # gamma=2
                # alpha=0.25
                # slot_loss = - alpha * (1 - pt) ** gamma * slot_label * tf.log(pt) - (1 - alpha) * pt ** gamma * (1 - slot_label) * tf.log(1 - pt)
                # slot_loss = tf.reduce_mean(slot_loss, name='slot_loss')
                slot_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=slot_label, logits=slot_logits)
                # slot_loss = slot_loss * tf.transpose([1.0, 10.0, 50.0, 15.0, 15.0, 15.0, 20.0, 1.0, 15.0, 12.0, 20.0, 25.0, 2.0])
                slot_loss = slot_loss * tf.transpose([1.0, 10.0, 250.0, 10.0, 10.0, 10.0, 15.0, 1.0, 10.0, 10.0, 20.0, 20.0, 2.0])
                slot_loss = tf.reduce_mean(slot_loss, name='slot_loss')
                
                print('slot_loss', slot_loss)

                slot_predict = tf.argmax(slot_logits, axis=-1, output_type=tf.int32)
                print('slot_predict', slot_predict)
                slot_correct_prediction = tf.equal(slot_label, slot_logits)
                slot_correct_prediction = tf.cast(slot_correct_prediction, tf.float32)
                slot_acc = tf.reduce_mean(slot_correct_prediction)
                print('slot_acc', slot_acc)
                print("slot is traingdebug ===========")
            else:
                slot_loss = tf.constant([0.0], dtype=tf.float32)
                slot_acc = tf.constant([0.0], dtype=tf.float32)
            slot_probabilities = tf.nn.sigmoid(slot_logits)
            slot_predict = tf.argmax(slot_logits, axis=-1, output_type=tf.int32)

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
            sg_probabilities = tf.math.sigmoid(sg_logits)
            sg_predict = tf.cast(tf.round((tf.sign(sg_logits) + 1) / 2),
                                 tf.int32)
                                 
        with tf.variable_scope("aa_loss"):

            # if mode == tf.estimator.ModeKeys.TRAIN:
            # output_layer_drop = tf.nn.dropout(
            #   output_layer, keep_prob=self.config.entity_dropout)
            aa_num_labels = self.config.max_seq_length
            aa_logits = tf.layers.dense(output_layer,
                                        aa_num_labels,
                                        name='aa_dense')
            aa_logits = tf.reshape(aa_logits,
                                   [-1, aa_num_labels * aa_num_labels])
            print(aa_logits)

            if mode != tf.estimator.ModeKeys.PREDICT:
                aa_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(aa_labels, tf.float32), logits=aa_logits)
                print(aa_logits)
                print(aa_loss)
                aa_loss = tf.reduce_sum(aa_loss, axis=-1)
                print(aa_loss)
                aa_loss = tf.reduce_mean(aa_loss, axis=-1)
                print(aa_loss)

                # sg_log_probs = tf.nn.log_softmax(sg_logits, axis=-1)
                # sg_one_hot_labels = tf.one_hot(sg_labels, depth=sg_num_labels, dtype=tf.float32)
                print(aa_logits)
                print(tf.argmax(aa_logits, -1))
                print(aa_labels)
                aa_predict = tf.cast(tf.round((tf.sign(aa_logits) + 1) / 2),
                                     tf.int32)
                aa_correct_prediction = tf.equal(tf.cast(aa_labels, tf.int32),
                                                 aa_predict)
                aa_correct_prediction = tf.cast(aa_correct_prediction,
                                                tf.float32)
                aa_acc = tf.reduce_mean(aa_correct_prediction)
                aa_acc = tf.reduce_mean(aa_acc)
                # print(sg_one_hot_labels)
                print("is traingdebug ===========")
            else:
                aa_loss = tf.constant([0.0], dtype=tf.float32)
                aa_acc = tf.constant([0.0], dtype=tf.float32)
            aa_probabilities = tf.math.sigmoid(aa_logits)
            aa_predict = tf.cast(tf.round((tf.sign(aa_logits) + 1) / 2),
                                 tf.int32)

        with tf.variable_scope("action_loss"):
            action_output_layer = tf.reshape(output_layer, [-1, hidden_size])
            action_logits = tf.matmul(action_output_layer,
                                      output_action_weight,
                                      transpose_b=True)
            if mode == tf.estimator.ModeKeys.TRAIN:
                action_logits = tf.nn.dropout(
                    action_logits, keep_prob=self.config.action_dropout)

            action_logits = tf.nn.bias_add(action_logits, output_action_bias)
            # intent_logits = tf.layers.dense(output_layer, 1)
            action_logits = tf.reshape(action_logits,
                                      [-1, self.config.max_seq_length])
            action_logits = tf.layers.dense(
                action_logits,
                action_num_labels,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.02),
                name="action_dense",
                use_bias=True)
            if mode != tf.estimator.ModeKeys.PREDICT:
                action_correct_prediction = tf.equal(
                    tf.cast(action_label, tf.int64),
                    tf.argmax(action_logits, -1))
                action_correct_prediction = tf.cast(action_correct_prediction,
                                                    tf.float32)
                action_acc = tf.reduce_mean(action_correct_prediction)

                action_one_hot_labels = tf.one_hot(action_label,
                                                   depth=action_num_labels,
                                                   dtype=tf.float32)
                action_loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=action_one_hot_labels, logits=action_logits)

                print(action_loss)
                action_loss = tf.reduce_mean(action_loss, name='action_loss')
                print("is traingdebug ===========")
            else:
                action_loss = tf.constant([0.0], dtype=tf.float32)
                action_acc = tf.constant([0.0], dtype=tf.float32)

            print(action_loss)

            action_probabilities = tf.nn.softmax(action_logits, axis=-1)
            action_predict = tf.argmax(action_probabilities, axis=-1)

        with tf.variable_scope("disambiguate_loss"):
            disambiguate_output_layer = tf.reshape(output_layer, [-1, hidden_size])
            disambiguate_logits = tf.matmul(disambiguate_output_layer,
                                        output_disambiguate_weight,
                                        transpose_b=True)
            if mode == tf.estimator.ModeKeys.TRAIN:
                disambiguate_logits = tf.nn.dropout(
                    disambiguate_logits, keep_prob=self.config.disambiguate_dropout)

            disambiguate_logits = tf.nn.bias_add(disambiguate_logits,
                                             output_disambiguate_bias)
            # classify_logits = tf.layers.dense(output_layer, 1)
            disambiguate_logits = tf.reshape(disambiguate_logits,
                                         [-1, self.config.max_seq_length])
            disambiguate_logits = tf.layers.dense(
                disambiguate_logits,
                disambiguate_num_labels,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.02),
                name="disambiguate_dense",
                use_bias=True)
            if mode != tf.estimator.ModeKeys.PREDICT:
                disambiguate_correct_prediction = tf.equal(
                    tf.cast(disambiguate_label, tf.int64),
                    tf.argmax(disambiguate_logits, -1))
                disambiguate_correct_prediction = tf.cast(
                    disambiguate_correct_prediction, tf.float32)
                disambiguate_acc = tf.reduce_mean(disambiguate_correct_prediction)

                disambiguate_one_hot_labels = tf.one_hot(disambiguate_label,
                                                     depth=disambiguate_num_labels,
                                                     dtype=tf.float32)
                disambiguate_loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=disambiguate_one_hot_labels, logits=disambiguate_logits)
                # disambiguate_loss = tf.nn.weighted_cross_entropy_with_logits(
                #     labels=disambiguate_one_hot_labels, logits=disambiguate_logits, pos_weight=2.0)
                    
                disambiguate_label_new = tf.round((tf.cast(tf.abs(disambiguate_label - 2),tf.float32) /1.5))
                print("disambiguate_label_new",disambiguate_label_new)
                print("disambiguate_loss",disambiguate_loss)
                # disambiguate_loss = tf.reduce_mean(disambiguate_loss,  axis=1)
                print("disambiguate_loss reduce",disambiguate_loss)

                disambiguate_loss = disambiguate_label_new * disambiguate_loss

                disambiguate_loss = tf.reduce_mean(disambiguate_loss,
                                               name='disambiguate_loss')
                print("is traingdebug ===========")
            else:
                disambiguate_loss = tf.constant([0.0], dtype=tf.float32)
                disambiguate_acc = tf.constant([0.0], dtype=tf.float32)

            print(disambiguate_loss)

            disambiguate_probabilities = tf.nn.softmax(disambiguate_logits, axis=-1)
            disambiguate_predict = tf.argmax(disambiguate_probabilities, axis=-1)
            
        with tf.variable_scope("from_system_loss"):
            from_system_output_layer = tf.reshape(output_layer, [-1, hidden_size])
            from_system_logits = tf.matmul(from_system_output_layer,
                                      output_from_system_weight,
                                      transpose_b=True)
            if mode == tf.estimator.ModeKeys.TRAIN:
                from_system_logits = tf.nn.dropout(
                    from_system_logits, keep_prob=self.config.from_system_dropout)

            from_system_logits = tf.nn.bias_add(from_system_logits, output_from_system_bias)
            # intent_logits = tf.layers.dense(output_layer, 1)
            from_system_logits = tf.reshape(from_system_logits,
                                      [-1, self.config.max_seq_length])
            from_system_logits = tf.layers.dense(
                from_system_logits,
                from_system_num_labels,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.02),
                name="from_system_dense",
                use_bias=True)
            print(from_system_logits)
            if mode != tf.estimator.ModeKeys.PREDICT:
                from_system_correct_prediction = tf.equal(
                    tf.cast(from_system_label, tf.int64),
                    tf.argmax(from_system_logits, -1))
                from_system_correct_prediction = tf.cast(from_system_correct_prediction,
                                                    tf.float32)
                from_system_acc = tf.reduce_mean(from_system_correct_prediction)

                from_system_one_hot_labels = tf.one_hot(from_system_label,
                                                  depth=from_system_num_labels,
                                                  dtype=tf.float32)
                # print(from_system_one_hot_labels)
                
                # from_system_loss = tf.nn.softmax_cross_entropy_with_logits(
                #     labels=from_system_one_hot_labels, logits=from_system_logits)
                    
                from_system_loss = tf.nn.weighted_cross_entropy_with_logits(
                    labels=from_system_one_hot_labels, logits=from_system_logits, pos_weight=2.0)
                from_system_label_new = tf.round((tf.cast(tf.abs(from_system_label - 2),tf.float32) /1.5))
                print("from_system_label_new",from_system_label_new)
                print("from_system_loss",from_system_loss)
                from_system_loss = tf.reduce_mean(from_system_loss,  axis=1)
                print("from_system_loss reduce",from_system_loss)

                from_system_loss = from_system_label_new * from_system_loss
                # from_system_loss = self.CE_with_prior(from_system_label, from_system_logits, [0.19723, 0.4833, 0.35444], from_system_num_labels)
                from_system_loss = tf.reduce_mean(from_system_loss, name='from_system_loss')
                # from_system_loss = self.test_softmax_focal_ce_3(from_system_num_labels, 2, 0.25, from_system_logits, from_system_one_hot_labels)
                print('weighted_cross_entropy_with_logits', from_system_loss)
                print("is traingdebug ===========")
            else:
                from_system_loss = tf.constant([0.0], dtype=tf.float32)
                from_system_acc = tf.constant([0.0], dtype=tf.float32)

            print(from_system_loss)

            from_system_probabilities = tf.nn.softmax(from_system_logits, axis=-1)
            from_system_predict = tf.argmax(from_system_probabilities, axis=-1)
            
        with tf.variable_scope("objects_num_loss"):
            objects_num_output_layer = tf.reshape(output_layer, [-1, hidden_size])
            objects_num_logits = tf.matmul(objects_num_output_layer,
                                      output_objects_num_weight,
                                      transpose_b=True)
            if mode == tf.estimator.ModeKeys.TRAIN:
                objects_num_logits = tf.nn.dropout(
                    objects_num_logits, keep_prob=self.config.objects_num_dropout)

            objects_num_logits = tf.nn.bias_add(objects_num_logits, output_objects_num_bias)
            # intent_logits = tf.layers.dense(output_layer, 1)
            objects_num_logits = tf.reshape(objects_num_logits,
                                      [-1, self.config.max_seq_length])
            objects_num_logits = tf.layers.dense(
                objects_num_logits,
                objects_num_num_labels,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.02),
                name="objects_num_dense",
                use_bias=True)
            print(objects_num_logits)
            if mode != tf.estimator.ModeKeys.PREDICT:
                objects_num_correct_prediction = tf.equal(
                    tf.cast(objects_num_label, tf.int64),
                    tf.argmax(objects_num_logits, -1))
                objects_num_correct_prediction = tf.cast(objects_num_correct_prediction,
                                                    tf.float32)
                objects_num_acc = tf.reduce_mean(objects_num_correct_prediction)

                objects_num_one_hot_labels = tf.one_hot(objects_num_label,
                                                  depth=objects_num_num_labels,
                                                  dtype=tf.float32)
                
                objects_num_loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=objects_num_one_hot_labels, logits=objects_num_logits)

                objects_num_loss = tf.reduce_mean(objects_num_loss, name='objects_num_loss')
                print('softmax_cross_entropy_with_logits', objects_num_loss)
                print("is traingdebug ===========")
            else:
                objects_num_loss = tf.constant([0.0], dtype=tf.float32)
                objects_num_acc = tf.constant([0.0], dtype=tf.float32)

            print(objects_num_loss)

            objects_num_probabilities = tf.nn.softmax(objects_num_logits, axis=-1)
            objects_num_predict = tf.argmax(objects_num_probabilities, axis=-1)

            
            # weight_ones = tf.ones(from_system_label.shape, dtype=tf.float32)
            # weight_zeros = tf.zeros(from_system_label.shape, dtype=tf.float32)
            # compare = tf.constant(2.0, shape=from_system_label.shape)
            # from_system_label = tf.to_float(from_system_label)
            # print("from_system_label", from_system_label, compare)
            # from_system_weight = tf.cond(from_system_label < compare, lambda: weight_ones, lambda: weight_zeros)
            
            # weight_ones = tf.ones(disambiguate_label.shape, dtype=tf.float32)
            # weight_zeros = tf.zeros(disambiguate_label.shape, dtype=tf.float32)
            # compare = tf.constant(2.0, shape=disambiguate_label.shape)
            # disambiguate_label = tf.to_float(disambiguate_label)
            # disambiguate_weight = tf.cond(disambiguate_label < compare, lambda: weight_ones, lambda: weight_zeros)

            total_loss = self.config.action_weight * action_loss + \
                self.config.disambiguate_weight * disambiguate_loss + \
                self.config.cx_weight * cx_loss + self.config.sg_weight * sg_loss + self.config.slot_weight * slot_loss + \
                self.config.from_system_weight* from_system_loss + \
                self.config.objects_num_weight * objects_num_loss
            print(total_loss)

            return (total_loss, action_loss, slot_loss, disambiguate_loss, from_system_loss,
                    disambiguate_probabilities, from_system_probabilities, disambiguate_predict, from_system_predict, action_probabilities,
                    slot_probabilities, action_predict, slot_predict,
                    disambiguate_acc, from_system_acc, action_acc, slot_acc, cx_loss, cx_acc, sg_loss,
                    sg_acc, objects_num_predict, objects_num_loss, objects_num_acc, objects_num_probabilities, aa_loss, aa_acc)

    def get_prediction_module(self, bert_model, features, is_training,
                              percent_done, mode):
        # n_classes = len(self._get_label_mapping())
        reprs = bert_model.get_sequence_output()
        # reprs = bert_model.get_pooled_output()

        # reprs = pretrain_helpers.gather_positions(
        #     reprs, tf.cast(features[self.name + "_entity_positions"],
        #                    tf.int32))
        action_num_labels = self.config.action_num_labels
        disambiguate_num_labels = self.config.disambiguate_num_labels
        from_system_num_labels = self.config.from_system_num_labels
        objects_num_num_labels = self.config.objects_num_num_labels
        slot_num_labels = self.config.slot_num_labels
        cixing_num_labels = self.config.cixing_num_labels

        # if self.config.modelsave_dir is not None:
        #     (losses, action_loss, slot_loss, disambiguate_loss,
        #         disambiguate_probabilities, disambiguate_predict, action_probabilities,
        #         slot_probabilities, action_predict, slot_predict,
        #         disambiguate_accuracy, action_accuracy, slot_accuracy, cx_loss, cx_acc, sg_loss,
        #      sg_acc) = self.create_model(
        #          bert_model, features[self.name + "_action"], action_num_labels,
        #          features[self.name + "_disambiguate"], disambiguate_num_labels,
        #          features[self.name + "_slot"], slot_num_labels, mode,
        #          features["input_ids"], features["input_ids"],
        #          features["input_ids"], cixing_num_labels, None)

        # else:
        print(features[self.name + "_slot"])
        (losses, action_loss, slot_loss, disambiguate_loss, from_system_loss,
            disambiguate_probabilities, from_system_probabilities, disambiguate_predict, from_system_predict, action_probabilities,
            slot_probabilities, action_predict, slot_predict,
            disambiguate_accuracy, from_system_accuracy, action_accuracy, slot_accuracy, cx_loss, cx_acc, sg_loss,
            sg_acc, objects_num_predict, objects_num_loss, objects_num_acc, objects_num_probabilities, aa_loss, aa_acc) = self.create_model(
             bert_model, features[self.name + "_action"], action_num_labels,
             features[self.name + "_disambiguate"], disambiguate_num_labels,
             features[self.name + "_slot"], slot_num_labels, mode,
             features["input_ids"], features[self.name + "_pos_tags"],
             features[self.name + "_pos_mask"], cixing_num_labels,
             features[self.name + "_sgnet_logits"], features[self.name + "_from_system"], from_system_num_labels, features[self.name + "_objects_num"], objects_num_num_labels, features[self.name + "_aanet_logits"], 2)
        # logits = tf.layers.dense(reprs, n_classes)
        # losses = tf.nn.softmax_cross_entropy_with_logits(
        #    labels=tf.one_hot(features[self.name + "_entity"], n_classes),
        #    logits=logits)
        # losses *= features[self.name + "_entity_mask"]
        # losses = tf.reduce_sum(losses, axis=-1)

        # if self.config.modelsave_dir is not None:
        #     return losses, dict(
        #         loss=losses,
        #         action_loss=action_loss,
        #         disambiguate_loss=disambiguate_loss,
        #         slot_loss=slot_loss,
        #         slot_probabilities=slot_probabilities,
        #         action_predict=action_predict,
        #         disambiguate_predict=disambiguate_predict,
        #         disambiguate_probabilities=disambiguate_probabilities,
        #         action_probabilities=action_probabilities,
        #         slot_predict=slot_predict,
        #         slot_accuracy=slot_accuracy,
        #         action_accuracy=action_accuracy,
        #         predictions=slot_predict,
        #         disambiguate_accuracy=disambiguate_accuracy,
        #         eid=features[self.name + "_eid"],
        #     )

        if mode != tf.estimator.ModeKeys.TRAIN:
            return losses, dict(
                slot_probabilities=slot_probabilities,
                action_predict=action_predict,
                action_probabilities=action_probabilities,
                slot_predict=slot_predict,
                disambiguate_predict=disambiguate_predict,
                from_system_predict=from_system_predict,
                objects_num_predict=objects_num_predict,
                disambiguate_probabilities=disambiguate_probabilities,
                from_system_probabilities=from_system_probabilities,
                objects_num_probabilities=objects_num_probabilities,
                predictions=slot_predict,
                slot_labels=features[self.name + "_slot"],
                action_labels=features[self.name + "_action"],
                disambiguate_labels=features[self.name + "_disambiguate"],
                from_system_labels=features[self.name + "_from_system"],
                objects_num_labels=features[self.name + "_objects_num"],
                eid=features[self.name + "_eid"],
            )
        else:
            return losses, dict(
                loss=losses,
                action_loss=action_loss,
                slot_loss=slot_loss,
                cx_loss=cx_loss,
                cx_acc=cx_acc,
                sg_loss=sg_loss,
                sg_acc=sg_acc,
                aa_loss=aa_loss,
                aa_acc=aa_acc,
                disambiguate_loss=disambiguate_loss,
                from_system_loss=from_system_loss,
                objects_num_loss=objects_num_loss,
                slot_probabilities=slot_probabilities,
                action_predict=action_predict,
                action_probabilities=action_probabilities,
                slot_predict=slot_predict,
                disambiguate_predict=disambiguate_predict,
                from_system_predict=from_system_predict,
                objects_num_predict=objects_num_predict,
                disambiguate_probabilities=disambiguate_probabilities,
                from_system_probabilities=from_system_probabilities,
                objects_num_probabilities=objects_num_probabilities,
                slot_accuracy=slot_accuracy,
                action_accuracy=action_accuracy,
                disambiguate_accuracy=disambiguate_accuracy,
                from_system_accuracy=from_system_accuracy,
                objects_num_accuracy=objects_num_acc,
                predictions=slot_predict,
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
