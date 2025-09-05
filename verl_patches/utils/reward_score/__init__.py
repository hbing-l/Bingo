# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source in ['openai/gsm8k', 'TIGER-Lab/TheoremQA']:
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval', 'Maxwell-Jia/AIME_2024', 'HuggingFaceH4/MATH-500']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
