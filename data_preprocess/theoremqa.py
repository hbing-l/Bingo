# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import datasets
from verl.utils.hdfs_io import copy, makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess TheoremQA dataset to Parquet")
    parser.add_argument(
        '--local_dir',
        type=str,
        default='dataset',
    )
    parser.add_argument(
        '--hdfs_dir',
        type=str,
        default=None,
    )
    args = parser.parse_args()

    data_source = 'TIGER-Lab/TheoremQA'
    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['test']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split_name):
        def process_fn(example, idx):
            if 'Picture' in example and example['Picture'] is not None:
                return None

            problem_raw = example.pop('Question')
            ground_truth = example.pop('Answer')

            prompt = problem_raw + ' ' + instruction_following

            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "user", "content": prompt}
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    "split": split_name,
                    "index": idx,
                    "raw_problem": problem_raw,
                }
            }
            return data
        return process_fn

    os.makedirs(args.local_dir, exist_ok=True)

    train_dataset = train_dataset.filter(lambda example: 'Picture' not in example or example['Picture'] is None)
    train_dataset = train_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
