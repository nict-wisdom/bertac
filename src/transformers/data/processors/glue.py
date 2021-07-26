# coding=utf-8
# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT). (Modifications for BERTAC)
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" GLUE processors and helpers """

""" Modified from glue.py for BERTAC implementation 
 The following function/classes have been Modified from 'glue.py' in the 
    original Huggingface Transformers for BERTAC implementation 
 function
    - def glue_convert_examples_to_features()
 DataProcessor classes
    - "cola": class ColaProcessor,
    - "mnli": class MnliProcessor,
    - "mnli-mm": class MnliMismatchedProcessor,
    - "mrpc": class MrpcProcessor,
    - "sst-2": class Sst2Processor,
    - "sts-b": class StsbProcessor,
    - "qqp": class QqpProcessor,
    - "qnli": class QnliProcessor,
    - "rte": class RteProcessor,
    - "wnli": class WnliProcessor,
"""

import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count

from ...file_utils import is_tf_available
from .utils import DataProcessor, InputExample4GLUE, InputFeatures4GLUE
import numpy as np
from tqdm import tqdm


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)

# Modified by Jong-Hoon Oh 
def glue_convert_examples_to_features(
    examples,
    tokenizer,
    cnn_stoi,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExample4GLUEs`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        cnn_stoi: String to index mapping table for CNN vocabularies 
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExample4GLUEs``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        # modified for handling BERTAC's input
        none_str = None
        stokens_upper = None
        stokens_upper2 = None
        qtokens_upper2 = tokenizer.tokenize_for_cnn(example.text_a, add_special_tokens=True)
        qtokens_upper = tokenizer.tokenize_for_cnn(example.text_c, add_special_tokens=True)
        if example.text_d is not None:
           stokens_upper = tokenizer.tokenize_for_cnn(example.text_d, add_special_tokens=True)
           stokens_upper2 = tokenizer.tokenize_for_cnn(example.text_b, add_special_tokens=True)
        inputs = tokenizer.encode_plus_for_cnn(text1=example.text_a, text_pair1=example.text_b, ftokens=none_str,
                                               ftokens_upper=qtokens_upper, cnn_stoi=cnn_stoi,
                                               stokens_upper=stokens_upper, 
                                               add_special_tokens=True, max_length=max_length,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        input_qids, input_pids    = inputs["input_qids"], inputs["input_pids"]
        inputs = tokenizer.encode_plus_for_cnn(text1=example.text_a, text_pair1=example.text_b, ftokens=none_str,
                                               ftokens_upper=qtokens_upper2, cnn_stoi=cnn_stoi,
                                               stokens_upper=stokens_upper2, 
                                               add_special_tokens=True, max_length=max_length,)
        input_qids2, input_pids2    = inputs["input_qids"], inputs["input_pids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        padding_qlength = max_length - len(input_qids)
        padding_plength = max_length - len(input_pids) if example.text_b is not None else 0
        padding_qlength2 = max_length - len(input_qids2)
        padding_plength2 = max_length - len(input_pids2) if example.text_b is not None else 0
        pad_token_aid=cnn_stoi['<pad>']
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            if padding_qlength > 0:
               input_qids = input_qids + ([pad_token_aid] * padding_qlength)
            else:
               input_qids = input_qids[:max_length]
            if input_pids is not None:
               if padding_plength > 0:
                  input_pids = input_pids + ([pad_token_aid] * padding_plength)
               else:
                  input_pids = input_pids[:max_length]
            if padding_qlength2 > 0:
               input_qids2 = input_qids2 + ([pad_token_aid] * padding_qlength2)
            else:
               input_qids2 = input_qids2[:max_length]
            if input_pids2 is not None:
               if padding_plength2 > 0:
                  input_pids2 = input_pids2 + ([pad_token_aid] * padding_plength2)
               else:
                  input_pids2 = input_pids2[:max_length]
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(input_qids) == max_length, "Error with input length {} vs {}".format(len(input_qids), max_length)
        if input_pids is not None:
           assert len(input_pids) == max_length, "Error with input length {} vs {}".format(len(input_pids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_qids: %s" % " ".join([str(x) for x in input_qids]))
            logger.info("input_qids2: %s" % " ".join([str(x) for x in input_qids2]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures4GLUE(
                uid=example.guid, input_ids=input_ids, input_qids=input_qids, input_pids=input_pids,
                input_qids2=input_qids2, input_pids2=input_pids2,
                attention_mask=attention_mask, token_type_ids=token_type_ids, label=label,
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features


# Modified by Jong-Hoon Oh
class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample4GLUE(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # added by Jong-Hoon Oh
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    # Modified by Jong-Hoon Oh
    def _create_examples(self, lines, set_type):
        """Creates MRPC examples for the training and dev sets."""
        """ Paraphrase tasks """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if set_type == "test":
               guid = "%s-%s" % (set_type, line[0])
               text_a = line[3]
               text_b = line[4]
               text_c = line[-2]
               text_d = line[-1]
               label = "0" 
            else:
               guid = "%s-%s" % (set_type, i)
               text_a = line[3]
               text_b = line[4]
               text_c = line[-2]
               text_d = line[-1]
               label = line[0]
            examples.append(InputExample4GLUE(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, text_d=text_d, label=label))
        return examples

# Modified by Jong-Hoon Oh
class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample4GLUE(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_dev_mm_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    # Added by Jong-Hoon Oh
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")

    # Added by Jong-Hoon Oh
    def get_test_mm_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")

    # Added by Jong-Hoon Oh
    def get_test_ax_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_ax.tsv")), "test_ax")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    # Modified by Jong-Hoon Oh
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        """ NLI Task """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            text_c = line[-2]
            text_d = line[-1]
            label = line[-3]
            examples.append(InputExample4GLUE(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, text_d=text_d, label=label))
        return examples

# Modified by Jong-Hoon Oh
class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    # Added by Jong-Hoon Oh
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")

# Modified by Jong-Hoon Oh
class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample4GLUE(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            tensor_dict["bsentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")


    # Added by Jong-Hoon Oh
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    # Modified by Jong-Hoon Oh
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        """ acceptability task (single sentence task)"""
        examples = []
        for (i, line) in enumerate(lines):
            if set_type == "test":
               guid = "%s-%s" % (set_type, line[0]) 
            else:
               guid = "%s-%s" % (set_type, i) 
            text_a = line[3]
            text_c = line[4]
            label = line[1]
            if set_type == "test":
               label = "0"
            examples.append(InputExample4GLUE(guid=guid, text_a=text_a, text_b=None, text_c=text_c, text_d=None, label=label))
        return examples

# Modified by Jong-Hoon Oh
class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample4GLUE(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # Added by Jong-Hoon Oh
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    # Modified by Jong-Hoon Oh
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        """ sentiment task (single sentence)"""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if set_type == "test":
               guid = "%s-%s" % (set_type, line[0])
               text_a = line[1]
               text_c = line[3]
               label = line[2]
            else:
               guid = "%s-%s" % (set_type, i)
               text_a = line[0]
               text_c = line[2]
               label = line[1]
            examples.append(InputExample4GLUE(guid=guid, text_a=text_a, text_b=None, text_c=text_c, text_d=None, label=label))
        return examples

# Modified by Jong-Hoon Oh
class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample4GLUE(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # Added by Jong-Hoon Oh
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    # Modified by Jong-Hoon Oh
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        """ sentence similarlity task """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            text_c = line[-2]
            text_d = line[-1]
            label  = line[-3]
            examples.append(InputExample4GLUE(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, text_d=text_d, label=label))
        return examples

# Modified by Jong-Hoon Oh
class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample4GLUE(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # Added by Jong-Hoon Oh
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    # Modified by Jong-Hoon Oh
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        """ paraphrase task """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                # some Qqp instances have no text_a, text_b, or label
                text_a = line[-5]
                text_b = line[-4]
                text_c = line[-2]
                text_d = line[-1]
                label = line[-3]
            except IndexError:
                continue 
            examples.append(InputExample4GLUE(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, text_d=text_d,label=label))
        return examples


# Modified by Jong-Hoon Oh
class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample4GLUE(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # Added by Jong-Hoon Oh
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    # Modified by Jong-Hoon Oh
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        """ QA/NLI task """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            text_c = line[-2]
            text_d = line[-1]
            label = line[-3]
            examples.append(InputExample4GLUE(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, text_d=text_d, label=label))
        return examples

# Modified by Jong-Hoon Oh
class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample4GLUE(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # Modified by Jong-Hoon Oh
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    # Modified by Jong-Hoon Oh
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        """ NLI task """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            text_c = line[-2]
            text_d = line[-1]
            label = line[-3]
            examples.append(InputExample4GLUE(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, text_d=text_d, label=label))
        return examples

# Modified by Jong-Hoon Oh
class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample4GLUE(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # Added by Jong-Hoon Oh
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    # Modified by Jong-Hoon Oh
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        """ coreference/NLI task """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            text_c = line[1]
            text_d = line[2]
            label = line[-1]
            examples.append(InputExample4GLUE(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, text_d=text_d, label=label))
        return examples


glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}
