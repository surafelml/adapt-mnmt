"""Custom hooks."""

from __future__ import print_function

import io
import time
import six
import json

import tensorflow as tf

from opennmt.utils import misc


class LogParametersCountHook(tf.train.SessionRunHook):
  """Simple hook that logs the number of trainable parameters."""

  def begin(self):
    tf.logging.info("Number of trainable parameters: %d", misc.count_parameters())


_DEFAULT_COUNTERS_COLLECTION = "counters"


def add_counter(name, tensor):
  """Registers a new counter.

  Args:
    name: The name of this counter.
    tensor: The integer ``tf.Tensor`` to count.

  Returns:
    An op that increments the counter.

  See Also:
    :meth:`opennmt.utils.misc.WordCounterHook` that fetches these counters
    to log their value in TensorBoard.
  """
  count = tf.cast(tensor, tf.int64)
  total_count_init = tf.Variable(
      initial_value=0,
      name=name + "_init",
      trainable=False,
      dtype=count.dtype)
  total_count = tf.assign_add(
      total_count_init,
      count,
      name=name)
  tf.add_to_collection(_DEFAULT_COUNTERS_COLLECTION, total_count)
  return total_count


class CountersHook(tf.train.SessionRunHook):
  """Hook that summarizes counters.

  Implementation is mostly copied from StepCounterHook.
  """

  def __init__(self,
               every_n_steps=100,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None,
               counters=None):
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError("exactly one of every_n_steps and every_n_secs should be provided.")
    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_steps,
        every_secs=every_n_secs)

    self._summary_writer = summary_writer
    self._output_dir = output_dir
    self._counters = counters

  def begin(self):
    if self._counters is None:
      self._counters = tf.get_collection(_DEFAULT_COUNTERS_COLLECTION)
    if not self._counters:
      return

    if self._summary_writer is None and self._output_dir:
      self._summary_writer = tf.summary.FileWriterCache.get(self._output_dir)

    self._last_count = [None for _ in self._counters]
    self._global_step = tf.train.get_global_step()
    if self._global_step is None:
      raise RuntimeError("Global step should be created to use WordCounterHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if not self._counters:
      return None
    return tf.train.SessionRunArgs([self._counters, self._global_step])

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    if not self._counters:
      return

    counters, step = run_values.results
    if self._timer.should_trigger_for_step(step):
      elapsed_time, _ = self._timer.update_last_triggered_step(step)
      if elapsed_time is not None:
        for i in range(len(self._counters)):
          if self._last_count[i] is not None:
            name = self._counters[i].name.split(":")[0]
            value = (counters[i] - self._last_count[i]) / elapsed_time
            if self._summary_writer is not None:
              summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
              self._summary_writer.add_summary(summary, step)
            tf.logging.info("%s: %g", name, value)
          self._last_count[i] = counters[i]


class LogPredictionTimeHook(tf.train.SessionRunHook):
  """Hooks that gathers and logs prediction times."""

  def begin(self):
    self._total_time = 0
    self._total_tokens = 0
    self._total_examples = 0

  def before_run(self, run_context):
    self._run_start_time = time.time()
    predictions = run_context.original_args.fetches
    return tf.train.SessionRunArgs(predictions)

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    self._total_time += time.time() - self._run_start_time
    predictions = run_values.results
    batch_size = next(six.itervalues(predictions)).shape[0]
    self._total_examples += batch_size
    length = predictions.get("length")
    if length is not None:
      if len(length.shape) == 2:
        length = length[:, 0]
      self._total_tokens += sum(length)

  def end(self, session):
    tf.logging.info("Total prediction time (s): %f", self._total_time)
    tf.logging.info("Average prediction time (s): %f", self._total_time / self._total_examples)
    if self._total_tokens > 0:
      tf.logging.info("Tokens per second: %f", self._total_tokens / self._total_time)


class SaveEvaluationPredictionHook(tf.train.SessionRunHook):
  """Hook that saves the evaluation predictions."""

  def __init__(self, model, output_file, post_evaluation_fn=None, predictions=None):
    """Initializes this hook.

    Args:
      model: The model for which to save the evaluation predictions.
      output_file: The output filename which will be suffixed by the current
        training step.
      post_evaluation_fn: (optional) A callable that takes as argument the
        current step and the file with the saved predictions.
      predictions: The predictions to save.
    """
    self._model = model
    self._output_file = output_file
    self._post_evaluation_fn = post_evaluation_fn
    self._predictions = predictions

  def begin(self):
    if self._predictions is None:
      self._predictions = misc.get_dict_from_collection("predictions")
    if not self._predictions:
      raise RuntimeError("The model did not define any predictions.")
    self._global_step = tf.train.get_global_step()
    if self._global_step is None:
      raise RuntimeError("Global step should be created to use SaveEvaluationPredictionHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.train.SessionRunArgs([self._predictions, self._global_step])

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    predictions, self._current_step = run_values.results
    self._output_path = "{}.{}".format(self._output_file, self._current_step)
    with io.open(self._output_path, encoding="utf-8", mode="a") as output_file:
      for prediction in misc.extract_batches(predictions):
        self._model.print_prediction(prediction, stream=output_file)

  def end(self, session):
    tf.logging.info("Evaluation predictions saved to %s", self._output_path)
    if self._post_evaluation_fn is not None:
      self._post_evaluation_fn(self._current_step, self._output_path)

"""
class LoadWeightsFromCheckpointHook(tf.train.SessionRunHook):
  #Hook that loads model variables from checkpoint before starting the training.

  def __init__(self, checkpoint_path, load_partial_weights):
    self.checkpoint_path = checkpoint_path

  def begin(self):
    var_list = tf.train.list_variables(self.checkpoint_path)

    names = []
    for name, _ in var_list:
        if not name.startswith("global_step") and not name.startswith("words_per_sec"):
          names.append(name)

    self.values = {}
    reader = tf.train.load_checkpoint(self.checkpoint_path)
    for name in names:
      self.values[name] = reader.get_tensor(name)

    tf_vars = []
    current_scope = tf.get_variable_scope()
    reuse = tf.AUTO_REUSE if hasattr(tf, "AUTO_REUSE") else True
    with tf.variable_scope(current_scope, reuse=reuse):
      for name, value in six.iteritems(self.values):
        tf_vars.append(tf.get_variable(name, shape=value.shape))

    self.placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    self.assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, self.placeholders)]

  def after_create_session(self, session, coord):
    for p, op, (_, value) in zip(self.placeholders, self.assign_ops, six.iteritems(self.values)):
      session.run(op, {p: value})
"""


class LoadWeightsFromCheckpointHook(tf.train.SessionRunHook):
  """"Hook that loads model variables from checkpoint before starting the training.
    Args: ?

    # TODO: loads partial weights of the initial model/net (encoder, decoder, embeddings, output_layer, or combination of these four)
          : replaces the previous hook base on the config options
    Returns:
  """
  def __init__(self, checkpoint_path, not_restore):
    self.checkpoint_path = checkpoint_path
    self.not_restore = not_restore

  def begin(self):
    var_list = tf.train.list_variables(self.checkpoint_path)

    #TODO: list comprehension for var_list not to restore, logging which vars are being restored and change in model params.
    #not_restore = self.partial_weights  # change to list []
    #names = [name for name, _ in var_list if name.split('/')[0] not in self.not_restore]

    #tf.logging.info("VARIABLE LIST IN MODEL: %s", var_list)
    names = []
    for name, _ in var_list: #must do heirarchical var adding, since remove,adding one affects the other
      # check only: optim & associated vars
      top_scope = name.split('/')[0]
      if (((top_scope == "optim") or (top_scope == "global_step") or (top_scope == "word_per_sec"))
              and (top_scope not in self.not_restore)):
        names.append(name)

      # check only enc-dec
      if (len(name.split('/')) >= 2):
        encdec_var = name.split('/')[1]

        #check if embs & projec should included with the enc-dec
        embs_proj = ""
        if (len(name.split('/')) >= 3):
          #embs_proj = '/'.join(name.split('/')[1:3])
          embs_proj = name.split('/')[2]

        if (((encdec_var == "encoder") or (encdec_var == "decoder") or (encdec_var == "shared_embeddings"))
                and (encdec_var not in self.not_restore)
                and (embs_proj != "w_embs")
                and (embs_proj != "dense")):
          names.append(name)

      # check only embs
      #if (len(name.split('/')) >= 3):
        embs_proj = '/'.join(name.split('/')[1:3])
        if (((embs_proj == "encoder/w_embs") or (embs_proj == "decoder/w_embs") or (embs_proj == "decoder/dense"))
                and (embs_proj not in self.not_restore)):
          names.append(name)

    #tf.logging.info("VARIABLE LIST TO RESTORE: %s", names) #TODO: JSON/YAML/PICKL LOG JUMP
    tf.logging.info("VARIABLE LIST TO RESTORE: %s", json.dumps(names, indent=2, sort_keys=True))
    #ValueError: Trying to share variable global_step, but specified dtype float32 and found dtype int64_ref.

    """
    names = [name for name, _ in var_list if name.split('/')[0] not in self.not_restore
             and (name.split('/')[0] == 'optim' and name.split('/')[0] not in self.not_restore)
             and (len(name.split('/')) == 2 and name.split('/')[1] not in self.not_restore)
             and (len(name.split('/')) >= 3 and '/'.join(name.split('/')[1:3]) not in self.not_restore)]
    """

    """
    for name, _ in var_list:
      if not name.startswith("optim"):
        if not name.startswith("global_step") and not name.startswith("words_per_sec"):
          names.append(name)
    """

    self.values = {}
    reader = tf.train.load_checkpoint(self.checkpoint_path)
    for name in names:
      tf.logging.info("READING TENSOR VALUES FOR VAR: %s", name)
      self.values[name] = reader.get_tensor(name)

    tf_vars = []
    current_scope = tf.get_variable_scope()
    reuse = tf.AUTO_REUSE if hasattr(tf, "AUTO_REUSE") else True
    with tf.variable_scope(current_scope, reuse=reuse):
      for name, value in six.iteritems(self.values):
        tf_vars.append(tf.get_variable(name, shape=value.shape)) #dtype=)

    self.placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    self.assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, self.placeholders)]

  def after_create_session(self, session, coord):
    for p, op, (_, value) in zip(self.placeholders, self.assign_ops, six.iteritems(self.values)):
      session.run(op, {p: value})


class VariablesInitializerHook(tf.train.SessionRunHook):
  """Hook that initializes some variables in the current session. This is useful
  when using internal variables (e.g. for value accumulation) that are not saved
  in the checkpoints.
  """

  def __init__(self, variables):
    """Initializes this hook.

    Args:
      variables: A list of variables to initialize.
    """
    self._variables = variables
    self._init_op = None

  def begin(self):
    self._init_op = tf.variables_initializer(self._variables)

  def after_create_session(self, session, coord):
    session.run(self._init_op)
