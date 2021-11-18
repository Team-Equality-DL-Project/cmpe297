# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TFX taxi template pipeline definition.

This file defines TFX pipeline and various components in the pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Optional, Text

import tensorflow as tf
from tfx import v1 as tfx

import tensorflow_model_analysis as tfma
import copy
from tfx.components import ImportExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
# from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.trainer import executor as trainer_executor
# from tfx.dsl.components.base import executor_spec
from tfx.components.base import executor_spec
# from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.components.trainer.executor import GenericExecutor
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import evaluator_pb2
from tfx.proto import example_gen_pb2

import pprint
import urllib
import absl
tf.get_logger().propagate = False
pp = pprint.PrettyPrinter()

from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

from ml_metadata.proto import metadata_store_pb2

_project_id = 'alzheimers-331518'
_gcp_region = 'us-west1-b'
_ai_platform_training_args = {
    'project': _project_id,
    'region': _gcp_region,
    # Starting from TFX 0.14, training on AI Platform uses custom containers:
    # https://cloud.google.com/ml-engine/docs/containers-overview
    # You can specify a custom container here. If not specified, TFX will use a
    # a public container image matching the installed version of TFX.
    # 'masterConfig': { 'imageUri': 'gcr.io/my-project/my-container' },
    # Note that if you do specify a custom container, ensure the entrypoint
    # calls into TFX's run_executor script (tfx/scripts/run_executor.py)
}

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
_ai_platform_serving_args = {
    'model_name': 'saved_model',
    'project_id': _project_id,
    # The region to use when serving the model. See available regions here:
    # https://cloud.google.com/ml-engine/docs/regions
    # Note that serving currently only supports a single region:
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.models#Model
    'regions': [_gcp_region],
}

def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    # TODO(step 7): (Optional) Uncomment here to use BigQuery as a data source.
    # query: Text,
    preprocessing_module_file: Text,
    run_module_file: Text,
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    eval_accuracy_threshold: float,
    serving_model_dir: Text,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    ai_platform_serving_args: Optional[Dict[Text, Any]] = None,
) -> pipeline.Pipeline:
  """Implements the chicago taxi pipeline with TFX."""

  components = []

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = ImportExampleGen(input_base=data_path)
  # TODO(step 7): (Optional) Uncomment here to use BigQuery as a data source.
  # example_gen = big_query_example_gen_component.BigQueryExampleGen(
  #     query=query)
  components.append(example_gen)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  # TODO(step 5): Uncomment here to add StatisticsGen to the pipeline.
  components.append(statistics_gen)

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True)
  # TODO(step 5): Uncomment here to add SchemaGen to the pipeline.
  components.append(schema_gen)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(  # pylint: disable=unused-variable
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])
  # TODO(step 5): Uncomment here to add ExampleValidator to the pipeline.
  components.append(example_validator)

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=preprocessing_module_file)
  components.append(transform)

  ai_platform_training_args = copy.copy(ai_platform_training_args)
  # Uses user-provided Python function that implements a model using TF-Learn.
    
  trainer_args = {
      # 'module_file': run_module_file,
      'examples': transform.outputs['transformed_examples'],
      'schema': schema_gen.outputs['schema'],
      'transform_graph': transform.outputs['transform_graph'],
      'train_args': train_args,
      'eval_args': eval_args,
      'custom_executor_spec': executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor),
  }
  if ai_platform_training_args is not None:
    trainer_args.update({
        'custom_executor_spec':
            executor_spec.ExecutorClassSpec(
                ai_platform_trainer_executor.GenericExecutor
            ),
        'custom_config': {
            ai_platform_trainer_executor.TRAINING_ARGS_KEY: ai_platform_training_args,
        }
    })

  trainer = Trainer(**trainer_args, module_file=run_module_file)
  components.append(trainer)

  # Get the latest blessed model for model validation.
  # model_resolver = ResolverNode(
  #     instance_name='latest_blessed_model_resolver',
  #     resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
  #     model=Channel(type=Model),
  #     model_blessing=Channel(type=ModelBlessing))
  # # TODO(step 6): Uncomment here to add ResolverNode to the pipeline.
  # components.append(model_resolver)
    
  model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')
  components.append(model_resolver)
    
  # Uses TFMA to compute evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compare to a baseline).
  eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='label_xf', model_type='tf_lite')], #tf_lite
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(
                class_name='SparseCategoricalAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.3}), # TODO: Change threshold if need be, right now it is very low.
                    # Change threshold will be ignored if there is no
                    # baseline model resolved from MLMD (first run).
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': -1e-3})))
        ])
    ])

  evaluator = tfx.components.Evaluator(
    examples=transform.outputs['transformed_examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)
  components.append(evaluator)
    
  pusher_args = {
      'model':
          trainer.outputs['model'],
      'push_destination':
          pusher_pb2.PushDestination(
              filesystem=pusher_pb2.PushDestination.Filesystem(
                  base_directory=serving_model_dir)),
  }
  if ai_platform_serving_args is not None:
    pusher_args.update({
        'custom_executor_spec':
            executor_spec.ExecutorClassSpec(ai_platform_pusher_executor.Executor),
        # 'custom_config': {
        #     ai_platform_pusher_executor.SERVING_ARGS_KEY: ai_platform_serving_args
        # },
    })
  pusher = Pusher(**pusher_args)
  components.append(pusher)

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      # Change this value to control caching of execution results. Default value
      # is `False`.
      # enable_cache=True,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args
  )
