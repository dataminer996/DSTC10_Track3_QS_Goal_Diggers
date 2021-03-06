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

"""Fine-tunes an ELECTRA model on a downstream task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import datetime
import tensorflow.compat.v1 as tf

import configure_finetuning
from finetune import preprocessing
from finetune import task_builder
from model import modeling
from model import optimization
from util import training_utils
from util import utils
import gzip
from sklearn import metrics
import os


class FinetuningModel(object):
  """Finetuning model with support for multi-task training."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tasks,
               is_training, features, num_train_steps, mode):
    # Create a shared transformer encoder
    bert_config = training_utils.get_bert_config(config)
    self.bert_config = bert_config
    if config.debug:
      bert_config.num_hidden_layers = 3
      bert_config.hidden_size = 144
      bert_config.intermediate_size = 144 * 4
      bert_config.num_attention_heads = 4
    assert config.max_seq_length <= bert_config.max_position_embeddings
    bert_model = modeling.BertModel(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=features["input_ids"],
        input_mask=features["input_mask"],
        token_type_ids=features["segment_ids"],
        use_one_hot_embeddings=config.use_tpu,
        embedding_size=config.embedding_size)
    percent_done = (tf.cast(tf.train.get_or_create_global_step(), tf.float32) /
                    tf.cast(num_train_steps, tf.float32))

    # Add specific tasks
    self.outputs = {"task_id": features["task_id"]}
    losses = []
    for task in tasks:
      with tf.variable_scope("task_specific/" + task.name):
        task_losses, task_outputs = task.get_prediction_module(
            bert_model, features, is_training, percent_done,mode)
        losses.append(task_losses)
        self.outputs[task.name] = task_outputs
        self.rasa_outputs = task_outputs

    self.loss = tf.reduce_sum(
        tf.stack(losses, -1) *
        tf.one_hot(features["task_id"], len(config.task_names)))


def model_fn_builder(config: configure_finetuning.FinetuningConfig, tasks,
                     num_train_steps, pretraining_config=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    utils.log("Building model...")
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = FinetuningModel(
        config, tasks, is_training, features, num_train_steps, mode)

    # Load pre-trained weights from checkpoint
    init_checkpoint = config.init_checkpoint
    utils.log("Using init checkpoint", init_checkpoint)
    utils.log("modeldir", pretraining_config)
    if pretraining_config is not None:
      latest_checkpoint = tf.train.latest_checkpoint(pretraining_config.model_dir)
      try:
          if latest_checkpoint:
            init_checkpoint = latest_checkpoint
      except:
          pass
      utils.log("Using checkpoint", init_checkpoint)
    tvars = tf.trainable_variables()
    scaffold_fn = None
    if init_checkpoint:
      assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
          tvars, init_checkpoint)
      print('===assignment_map', assignment_map)
      if config.use_tpu:
        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()
        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # Build model for training or prediction
    if mode == tf.estimator.ModeKeys.TRAIN:
      print("========++==========",model.loss)
      train_op = optimization.create_optimizer(
          model.loss, config.learning_rate, num_train_steps,
          weight_decay_rate=config.weight_decay_rate,
          use_tpu=config.use_tpu,
          warmup_proportion=config.warmup_proportion,
          layerwise_lr_decay_power=config.layerwise_lr_decay,
          n_transformer_layers=model.bert_config.num_hidden_layers
      )
      print(model.rasa_outputs)
    #   hook_dict = {}
    #   hook_dict['loss'] = model.rasa_outputs['loss']
    #   hook_dict['action_loss'] = model.rasa_outputs['action_loss']
    #   hook_dict['disambiguate_loss'] = model.rasa_outputs['disambiguate_loss']
    #   hook_dict['slot_loss'] = model.rasa_outputs['slot_loss']
    #   hook_dict['cx_loss'] = model.rasa_outputs['cx_loss']
    #   hook_dict['cx_acc'] = model.rasa_outputs['cx_acc']
    #   hook_dict['sg_loss'] = model.rasa_outputs['sg_loss']
    #   hook_dict['sg_acc'] = model.rasa_outputs['sg_acc']

    #   hook_dict['action_accuracy'] = model.rasa_outputs['action_accuracy']
    #   hook_dict['disambiguate_accuracy'] = model.rasa_outputs['disambiguate_accuracy']
    #   hook_dict['slot_accuracy'] = model.rasa_outputs['slot_accuracy']
    #   hook_dict['global_steps'] = tf.train.get_or_create_global_step()

      # action_accuracy=tf.summary.scalar('action_accuracy', model.rasa_outputs['action_accuracy'])
      # action_loss=tf.summary.scalar('action_loss', model.rasa_outputs['action_loss'])
      # disambiguate_accuracy=tf.summary.scalar('disambiguate_accuracy', model.rasa_outputs['disambiguate_accuracy'])
      # disambiguate_loss=tf.summary.scalar('disambiguate_loss', model.rasa_outputs['disambiguate_loss'])
      # slot_accuracy=tf.summary.scalar('slot_accuracy', model.rasa_outputs['slot_accuracy'])
      # slot_loss=tf.summary.scalar('slot_loss', model.rasa_outputs['slot_loss'])

      # cx_acc=tf.summary.scalar('cx_acc', model.rasa_outputs['cx_acc'])
      # cx_loss=tf.summary.scalar('cx_loss', model.rasa_outputs['cx_loss'])

      # sg_acc=tf.summary.scalar('sg_acc', model.rasa_outputs['sg_acc'])
      # sg_loss=tf.summary.scalar('sg_loss', model.rasa_outputs['sg_loss'])

     # summary_op = tf.summary.merge_all()
     # summary_train_hook = tf.train.SummarySaverHook(
     #           save_steps=200,
     #           output_dir=config.model_dir,
      #          summary_op=summary_op)
    #   logging_hook = tf.train.LoggingTensorHook(
    #             hook_dict, every_n_iter=10)
    #   print(config.use_tpu)        
      
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=model.loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          training_hooks=[training_utils.ETAHook(
              {} if config.use_tpu else dict(loss=model.loss),
              num_train_steps, config.iterations_per_loop, config.use_tpu, 10)])
      #output_spec = tf.estimator.tpu.TPUEstimatorSpec(
      #   mode=mode,
      #    loss=model.loss,
      #    train_op=train_op,
      #    scaffold_fn=scaffold_fn,
      #    training_hooks=[logging_hook,summary_train_hook,training_utils.ETAHook(
      #        {} if config.use_tpu else dict(loss=model.loss),
      #        num_train_steps, config.iterations_per_loop, config.use_tpu, 10)])
    else:
#      assert mode == tf.estimator.ModeKeys.PREDICT
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=utils.flatten_dict(model.outputs),
          scaffold_fn=scaffold_fn)
      if config.modelsave_dir is not None: 
          output_spec = tf.estimator.EstimatorSpec(
             mode=mode, predictions=utils.flatten_dict(model.outputs)
          )

    utils.log("Building complete")
    return output_spec

  return model_fn


class ModelRunner(object):
  """Fine-tunes a model on a supervised task."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tasks,
               pretraining_config=None):
    self._config = config
    self._tasks = tasks
    self._preprocessor = preprocessing.Preprocessor(config, self._tasks)

    is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    tpu_cluster_resolver = None
    if config.use_tpu and config.tpu_name:
      tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          config.tpu_name, zone=config.tpu_zone, project=config.gcp_project)
    tpu_config = tf.estimator.tpu.TPUConfig(
        iterations_per_loop=config.iterations_per_loop,
        num_shards=config.num_tpu_cores,
        per_host_input_for_training=is_per_host,
        tpu_job_name=config.tpu_job_name)
    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=config.model_dir,
        save_checkpoints_steps=config.save_checkpoints_steps,
        save_checkpoints_secs=None,
        tpu_config=tpu_config)

    if self._config.do_train:
      (self._train_input_fn,
       self.train_steps) = self._preprocessor.read_train()
    else:
      self._train_input_fn, self.train_steps = None, 0
    # print('===train steps: ', self.train_steps)
    model_fn = model_fn_builder(
        config=config,
        tasks=self._tasks,
        num_train_steps=self.train_steps,
        pretraining_config=pretraining_config)
    if config.modelsave_dir is not None: 
        print('=======================')   
        self._estimator =tf.estimator.Estimator(
           model_fn=model_fn,
           config=run_config)
    else:
        print('--------------------')
        self._estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=config.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        predict_batch_size=config.eval_batch_size)

  def train(self):
    utils.log("Training for {:} steps".format(self.train_steps))
    self._estimator.train(
        input_fn=self._train_input_fn, max_steps=self.train_steps)

  def evaluate(self):
    return {task.name: self.evaluate_task(task) for task in self._tasks}
    
  def predict(self, split):
    return {task.name: self.predict_task(task, split) for task in self._tasks}
    # return {task.name: self.evaluate_task(task, split) for task in self._tasks}

  def evaluate_task(self, task, split="dev", return_results=True):
    """Evaluate the current model."""
    utils.log("Evaluating", task.name, split)
    eval_input_fn, _ = self._preprocessor.read_predict([task], split)
    if tf.gfile.Exists(os.path.join(self._config.init_checkpoint, 'step1_eval_result.csv')):
    # if os.path.exists(os.path.join('./data/learning_rate_record.csv', 'learning_rate_record.csv')):
        # output = open('./data/learning_rate_record.csv', 'a+') 
        output = tf.io.gfile.GFile(os.path.join(self._config.init_checkpoint, 'step1_eval_result.csv'), mode='a+')
    else:
        # output = open('./data/learning_rate_record.csv', 'w+') 
        output = tf.io.gfile.GFile(os.path.join(self._config.init_checkpoint, 'step1_eval_result.csv'), mode='w+')
        output.write('train_name,modeldir_name,action_f1,disambiguate_f1,from_system_f1,slot_f1,objects_num_f1\n')
        
    if tf.gfile.Exists(os.path.join(self._config.model_dir, 'step1_eval_result_detail.txt')):
    # if os.path.exists(os.path.join('./data/learning_rate_record.csv', 'learning_rate_record.csv')):
        # output = open('./data/learning_rate_record.csv', 'a+') 
        detail_output = tf.io.gfile.GFile(os.path.join(self._config.model_dir, 'step1_eval_result_detail.txt'), mode='a+')
    else:
        # output = open('./data/learning_rate_record.csv', 'w+') 
        detail_output = tf.io.gfile.GFile(os.path.join(self._config.model_dir, 'step1_eval_result_detail.txt'), mode='w+')
    # print(eval_input_fn)
   # results = self._estimator.evaluate(input_fn=eval_input_fn,
    #                                  yield_single_examples=True)
    # results = self._estimator.evaluate(input_fn=eval_input_fn)
    results = self._estimator.predict(input_fn=eval_input_fn,yield_single_examples=True)
    # print('predict result', results)
    
    action_labels = []
    action_predict = []
    disambiguate_labels = []
    disambiguate_predict = []
    from_system_labels = []
    from_system_predict = []
    objects_num_labels = []
    objects_num_predict = []
    slot_labels = []
    slot_predict = []
    count = 0
    for result in results:
        if count%1000 == 0:
          print(count)
        # print(result)
        # print('%%%%%%%%result predict', result['chunk_slot_labels'], result['chunk_slot_predict'])
        action_labels.append(result['chunk_action_labels'])
        action_predict.append(result['chunk_action_predict'])
        
        if int(result['chunk_disambiguate_labels']) != 2:
            disambiguate_labels.append(result['chunk_disambiguate_labels'])
            disambiguate_predict.append(result['chunk_disambiguate_predict'])
            
        if int(result['chunk_from_system_labels']) != 2:
            from_system_labels.append(result['chunk_from_system_labels'])
            from_system_predict.append(result['chunk_from_system_predict'])
            
        objects_num_labels.append(result['chunk_objects_num_labels'])
        objects_num_predict.append(result['chunk_objects_num_predict'])
        
        slot_labels.append(result['chunk_slot_labels'])
        slot_pred = []
        for e in result['chunk_slot_probabilities']:
          if e > 0.5:
            slot_pred.append(1)
          else:
            slot_pred.append(0)
        slot_predict.append(slot_pred)
        count += 1
    
    action_result = metrics.classification_report(action_labels,action_predict,target_names=None,digits=4)
    disambiguate_result = metrics.classification_report(disambiguate_labels,disambiguate_predict,target_names=None,digits=4)
    from_system_result = metrics.classification_report(from_system_labels,from_system_predict,target_names=None,digits=4)
    slot_result = metrics.classification_report(slot_labels,slot_predict,target_names=None,digits=4)
    objects_num_result = metrics.classification_report(objects_num_labels,objects_num_predict,target_names=None,digits=4)

    total = 0
    error = 0
    for i  in  range(len(action_labels)):             
        #  if action_labels[i] == 0:
        #      continue  
         total = total + 1
         if action_labels[i] != action_predict[i]:
            error = error + 1
            
    action_acc = (total - error) / total

    total = 0
    error = 0
    for i  in  range(len(disambiguate_labels)):
         total = total + 1
         if disambiguate_labels[i] != disambiguate_predict[i]:
            error = error + 1

    disambiguate_acc = (total - error) / total
    
    total = 0
    error = 0
    for i  in  range(len(from_system_labels)):
         total = total + 1
         if from_system_labels[i] != from_system_predict[i]:
            error = error + 1

    from_system_acc = (total - error) / total
    
    total = 0
    error = 0
    for i  in  range(len(objects_num_labels)):
         total = total + 1
         if objects_num_labels[i] != objects_num_predict[i]:
            error = error + 1

    objects_num_acc = (total - error) / total
   
    total = 0
    error = 0
    for i in range(len(slot_labels)):
        # print(len(slot_labels[i]))
        for j in range(len(slot_labels[i])): 
          if slot_labels[i][j] == 1:
              total = total + 1
              if not slot_labels[i][j] != slot_predict[i][j]:
                  error = error + 1
    
    slot_acc = (total - error) / total          
    
    print(action_result)
    print(disambiguate_result)
    print(from_system_result)
    print(slot_result)
    print(objects_num_result)
    
    print(action_acc)
    print(disambiguate_acc)
    print(from_system_acc)
    print(slot_acc)
    print(objects_num_acc)
    
    action_f1 = metrics.f1_score(action_labels,action_predict,average='weighted')
    disambiguate_f1 = metrics.f1_score(disambiguate_labels,disambiguate_predict,average='weighted')
    from_system_f1 = metrics.f1_score(from_system_labels,from_system_predict,average='weighted')
    slot_f1 = metrics.f1_score(slot_labels,slot_predict,average='weighted')
    objects_num_f1 = metrics.f1_score(objects_num_labels,objects_num_predict,average='weighted')
    
    weight_dict = {'action_weight': self._config.action_weight,'disambiguate_weight': self._config.disambiguate_weight,'from_system_weight': self._config.from_system_weight,\
      'slot_weight': self._config.slot_weight, 'cx_weight':self._config.cx_weight, 'sg_weight':self._config.sg_weight, 'objects_num_weight': self._config.objects_num_weight,\
        'aa_weight': self._config.aa_weight
    }
    train_name = 'learning_rate_' + str(self._config.learning_rate) + '_epoch_' + str(self._config.num_train_epochs)
    modeldir_name = self._config.model_dir
    for weight_name, weight_value in weight_dict.items():
        if weight_value > 0.0:
            train_name += '_' + weight_name + '_' + str(weight_value)
    output.write(','.join([train_name, modeldir_name, str(action_f1), str(disambiguate_f1), str(from_system_f1), str(slot_f1), str(objects_num_f1)]) + '\n')
    output.close()
    detail_output.write('=============\n' + modeldir_name + '\n action_result\n' + action_result + '\ndisambiguate_result\n' + disambiguate_result +  '\nfrom_system_result\n' + from_system_result +  '\nslot_result\n' + slot_result +  '\nobjects_num_result\n' + objects_num_result + '\n')
    detail_output.close()

    return 

  def predict_task(self, task, split="test", return_results=True):
    """Evaluate the current model."""
    utils.log("Predicting", task.name)
    pred_input_fn, _ = self._preprocessor.read_predict([task], split)
    results = self._estimator.predict(input_fn=pred_input_fn, yield_single_examples=True)
    print('predict result111', results)
    #output_eval_file = os.path.join(self._config.model_dir, "predict_results.txt")

    weight_dict = {'action_weight': self._config.action_weight,'disambiguate_weight': self._config.disambiguate_weight,'from_system_weight': self._config.from_system_weight,\
      'slot_weight': self._config.slot_weight, 'cx_weight':self._config.cx_weight, 'sg_weight':self._config.sg_weight, 'objects_num_weight': self._config.objects_num_weight,\
        'aa_weight': self._config.aa_weight
    }
    train_name = os.path.basename(self._config.model_dir) + '_pred_result.txt'
    #path to save prediction
    output_eval_file = os.path.join(self._config.save_prediction, train_name)
    writer = open(output_eval_file, "w+")
    writer = tf.io.gfile.GFile(output_eval_file, mode='w+')
    print('Writing ', output_eval_file)
    
    # input_eval_file = os.path.join(self._config.raw_data_dir(task.name), split + ".txt")
    # input_txt = open(input_eval_file, 'r').readlines()
    
    action_labels = []
    action_predict = []
    disambiguate_labels = []
    disambiguate_predict = []
    from_system_labels = []
    from_system_predict = []
    slot_labels = []
    slot_predict = []
    objects_num_labels = []
    objects_num_predict = []
    count = 0
    for result in results:
        if count%100 == 0:
          print(count)
        # print(result)
        result_json = {}
        result_json["action"] = str(result['chunk_action_predict'])
        result_json["action_prob"] = [str(each) for each in result['chunk_action_probabilities']]
        result_json["disambiguate"] = str(result['chunk_disambiguate_predict'])
        result_json["disambiguate_prob"] = [str(each) for each in result['chunk_disambiguate_probabilities']]
        result_json["from_system"] = str(result['chunk_from_system_predict'])
        result_json["from_system_prob"] = [str(each) for each in result['chunk_from_system_probabilities']]
        result_json["objects_num"] = str(result['chunk_objects_num_predict'])
        result_json["objects_num_prob"] = [str(each) for each in result['chunk_objects_num_probabilities']]

        result_json["eid"] = str(result['chunk_eid'])

        slot_pred = []
        for e in result['chunk_slot_probabilities']:
          if e > 0.5:
            slot_pred.append('1')
          else:
            slot_pred.append('0')

        result_json["slot"] = list(slot_pred)
        result_json["slot_prob"] = [str(each) for each in result['chunk_slot_probabilities']]
        count += 1
        
        action_labels.append(result['chunk_action_labels'])
        action_predict.append(result['chunk_action_predict'])
        disambiguate_labels.append(result['chunk_disambiguate_labels'])
        disambiguate_predict.append(result['chunk_disambiguate_predict'])
        from_system_labels.append(result['chunk_from_system_labels'])
        from_system_predict.append(result['chunk_from_system_predict'])
        slot_labels.append(result['chunk_slot_labels'])
        slot_predict.append([int(each) for each in slot_pred])
        objects_num_labels.append(result['chunk_objects_num_labels'])
        objects_num_predict.append(result['chunk_objects_num_predict'])

        writer.write(json.dumps(result_json) + '\n')
        # if not slot_pred == result['chunk_slot_labels']:
        #     print(input_txt[count])
        #     print(result['chunk_slot_labels'])
        #     print(slot_pred)
        
            
    writer.close()
    action_result = metrics.classification_report(action_labels,action_predict,target_names=None,digits=4)
    disambiguate_result = metrics.classification_report(disambiguate_labels,disambiguate_predict,target_names=None,digits=4)
    from_system_result = metrics.classification_report(from_system_labels,from_system_predict,target_names=None,digits=4)
    objects_num_result = metrics.classification_report(objects_num_labels,objects_num_predict,target_names=None,digits=4)
    slot_result = metrics.classification_report(slot_labels,slot_predict,target_names=None,digits=4)
    print(action_result)
    print(disambiguate_result)
    print(from_system_result)
    print(slot_result)
    print(objects_num_result)
        
    return

  def write_classification_outputs(self, tasks, trial, split):
    """Write classification predictions to disk."""
    utils.log("Writing out predictions for", tasks, split)
    predict_input_fn, _ = self._preprocessor.prepare_predict(tasks, split)
    results = self._estimator.predict(input_fn=predict_input_fn,
                                      yield_single_examples=True)
    # task name -> eid -> model-logits
    logits = collections.defaultdict(dict)
    for r in results:
      if r["task_id"] != len(self._tasks):
        r = utils.nest_dict(r, self._config.task_names)
        task_name = self._config.task_names[r["task_id"]]
        logits[task_name][r[task_name]["eid"]] = (
            r[task_name]["logits"] if "logits" in r[task_name]
            else r[task_name]["predictions"])
    for task_name in logits:
      utils.log("Pickling predictions for {:} {:} examples ({:})".format(
          len(logits[task_name]), task_name, split))
      if trial <= self._config.n_writes_test:
        utils.write_pickle(logits[task_name], self._config.test_predictions(
            task_name, split, trial))


def write_results(config: configure_finetuning.FinetuningConfig, results):
  """Write evaluation metrics to disk."""
  utils.log("Writing results to", config.results_txt)
  utils.mkdir(config.results_txt.rsplit("/", 1)[0])
  utils.write_pickle(results, config.results_pkl)
  with tf.io.gfile.GFile(config.results_txt, "w") as f:
    results_str = ""
    for trial_results in results:
      for task_name, task_results in trial_results.items():
        if task_name == "time" or task_name == "global_step":
          continue
        results_str += task_name + ": " + " - ".join(
            ["{:}: {:.2f}".format(k, v)
             for k, v in task_results.items()]) + "\n"
    f.write(results_str)
  utils.write_pickle(results, config.results_pkl)


def run_finetuning(config: configure_finetuning.FinetuningConfig):
  """Run finetuning."""

  # Setup for training
  results = []
  trial = 1
  heading_info = "model={:}, trial {:}/{:}".format(
      config.model_name, trial, config.num_trials)
  heading = lambda msg: utils.heading(msg + ": " + heading_info)
  heading("Config")
  utils.log_config(config)
  generic_model_dir = config.model_dir
  tasks = task_builder.get_tasks(config)
  if config.todo_task == 'finetune':
    print("start run_finetuning ")

    # Train and evaluate num_trials models with different random seeds
    while config.num_trials < 0 or trial <= config.num_trials:
      #config.model_dir = generic_model_dir + "_" + str(trial)
      #if config.do_train:
      #  utils.rmkdir(config.model_dir)

    #   model_runner = ModelRunner(config, tasks)
      model_runner = ModelRunner(config, tasks, config)
      
      if config.modelsave_dir is not None:         
          feature_columns = [tf.feature_column.numeric_column(key='input_ids', shape=(config.max_seq_length,), default_value=None, dtype=tf.int64, normalizer_fn=None),
          tf.feature_column.numeric_column(key='input_mask', shape=(config.max_seq_length,), default_value=None, dtype=tf.int64, normalizer_fn=None),
          tf.feature_column.numeric_column(key='segment_ids', shape=(config.max_seq_length,), default_value=None, dtype=tf.int64, normalizer_fn=None),
          tf.feature_column.numeric_column(key='task_id',shape=(), default_value=None, dtype=tf.int64, normalizer_fn=None),
          tf.feature_column.numeric_column(key='chunk_action',shape=(), default_value=None, dtype=tf.int64, normalizer_fn=None),
          tf.feature_column.numeric_column(key='chunk_disambiguate',shape=(), default_value=None, dtype=tf.int64, normalizer_fn=None),
          tf.feature_column.numeric_column(key='chunk_slot',shape=(config.slot_num_labels,), default_value=None, dtype=tf.int64, normalizer_fn=None),
          tf.feature_column.numeric_column(key='chunk_eid',shape=(), default_value=None, dtype=tf.int64, normalizer_fn=None)]

          feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
          print(feature_spec)  
          serving_input_receiver_fn =  tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
          print("start save model....")
          print("pbfile save dir:",config.modelsave_dir)
          model_runner._estimator.export_saved_model(config.modelsave_dir, serving_input_receiver_fn)
          print("modelsave finish")
        #   exit(1)
          
      if config.do_train:
        heading("Start training")
        model_runner.train()
        utils.log()

      if config.do_eval:
        heading("Run dev set evaluation")
        model_runner.evaluate()
        utils.log()
      
      if config.do_predict:
        heading("Run test set prediction")
        model_runner.predict(config.do_predict_split)
        utils.log()
      trial += 1
        
  elif config.todo_task == 'feature':
    print("start featuring ")
    preprocessor = preprocessing.Preprocessor(config, tasks)

    if config.do_train:
      preprocessor.prepare_train()

    if config.do_eval:
      preprocessor.prepare_predict(tasks, 'dev')
    
    if config.do_predict:
      preprocessor.prepare_predict(tasks, config.do_predict_split)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", required=True,
                      help="Location of data files (model weights, etc).")
  parser.add_argument("--model-name", required=True,
                      help="The name of the model being fine-tuned.")
  parser.add_argument("--todo-task", required=True,
                      help="finetun or feature.")
  parser.add_argument("--model-dir", required=True,
                      help="path to save model")
  parser.add_argument("--do-predict-split", required=True,
                      help="test or devtest")
  parser.add_argument("--use-sgnet", required=True,
                      help="use sgnet or not")
  parser.add_argument("--hparams", default="{}",
                      help="JSON dict of model hyperparameters.")
  args = parser.parse_args()
  if args.hparams.endswith(".json"):
    hparams = utils.load_json(args.hparams)
  else:
    hparams = json.loads(args.hparams)
  tf.logging.set_verbosity(tf.logging.ERROR)
  run_finetuning(configure_finetuning.FinetuningConfig(
      args.model_name, args.data_dir, args.todo_task, args.model_dir, args.do_predict_split, args.use_sgnet, **hparams))


if __name__ == "__main__":
  main()
