# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


DATA_ROOT = os.environ.get("BAGEL_DATA_ROOT", "your_data_path/bagel_example")


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': os.path.join(DATA_ROOT, 't2i'), # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': os.path.join(DATA_ROOT, 'editing/seedxedit_multi'),
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": os.path.join(DATA_ROOT, 'editing/parquet_info/seedxedit_multi_nas.json'), # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': os.path.join(DATA_ROOT, 'vlm/images'),
			'jsonl_path': os.path.join(DATA_ROOT, 'vlm/llava_ov_si.jsonl'),
			'num_total_samples': 1000
		},
    },
}
