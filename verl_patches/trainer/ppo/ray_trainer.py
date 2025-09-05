# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Update AdvantageEstimator
class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    BINGO = "bingo"


# Add normalization function
def normalize_probs(probs_list):
    normalized_probs = []
    for chunk_probs in probs_list:
        chunk_probs = np.array(chunk_probs)
        min_val = np.min(chunk_probs)
        max_val = np.max(chunk_probs)
        if max_val == min_val:
            normalized_chunk = np.ones_like(chunk_probs)
        else:
            normalized_chunk = (chunk_probs - min_val) / (max_val - min_val)
        normalized_probs.append(normalized_chunk)
    return normalized_probs


# Compute advantages using the specified estimator of Bingo
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, lambda_c=1.0, lambda_w_n=1.0, lambda_w_e=1.0, tokenizer=None, i_threshold=0.3, initial_slope=None, dynamic_lambda_w_e=None, ppo=False):
    if adv_estimator == AdvantageEstimator.BINGO:

        # Import the compress_prompt function
        from compressor_client import compress_prompt

        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        old_log_probs = data.batch['old_log_probs']
        ref_responses = data.batch['ref_responses']
                
        current_lengths = response_mask.sum(dim=1)  # [batch_size]
        ref_response_mask = data.batch['ref_attention_mask'][:, -token_level_rewards.size(1):]
        ref_lengths = ref_response_mask.sum(dim=1)  # [batch_size]

        ratio_term = (current_lengths / ref_lengths).unsqueeze(1)  # [batch_size, 1]
        
        ratio_term_token = torch.zeros_like(token_level_rewards)  # [batch_size, seq_len]
        
        all_normalized_probs = []
        all_total_tokens = []
        all_low_prob_tokens = []
        all_high_prob_tokens = []

        ref_normalized_probs = []
        ref_total_tokens = []
        ref_low_prob_tokens = []
        ref_high_prob_tokens = []

        print("\n===== Statistics for original responses =====")

        for i, (response, length) in enumerate(zip(responses, current_lengths)):
            if length > 0:
                valid_tokens = response[:length]
                text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                
                result = compress_prompt(text, 0.6)

                
                sample_original_probs = result["original_probs"]
                sample_normalized_probs = normalize_probs(sample_original_probs)
                
                for probs in sample_normalized_probs:
                    total_tokens = len(probs)
                    low_prob_tokens = len([p for p in probs if p < i_threshold])
                    high_prob_tokens = total_tokens - low_prob_tokens
                    
                    all_normalized_probs.append(probs)
                    all_total_tokens.append(total_tokens)
                    all_low_prob_tokens.append(low_prob_tokens)
                    all_high_prob_tokens.append(high_prob_tokens)
                    
                    print(f"Original Samples {i+1}: Total tokens: {total_tokens}, Low prob: {low_prob_tokens} ({low_prob_tokens/total_tokens*100:.2f}%), High prob: {high_prob_tokens} ({high_prob_tokens/total_tokens*100:.2f}%)")
            else:
                all_normalized_probs.append([])
                all_total_tokens.append(0)
                all_low_prob_tokens.append(0)
                all_high_prob_tokens.append(0)
                print(f"Original Samples {i+1}: Empty sequence")

        print("\n===== Statistics for reference responses =====")
        for i, (ref_response, length) in enumerate(zip(ref_responses, ref_lengths)):
            if length > 0:
                valid_tokens = ref_response[:length]
                text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                
                result = compress_prompt(text, 0.6)

                sample_original_probs = result["original_probs"]
                sample_normalized_probs = normalize_probs(sample_original_probs)
                
                for probs in sample_normalized_probs:
                    total_tokens = len(probs)
                    low_prob_tokens = len([p for p in probs if p < i_threshold])
                    high_prob_tokens = total_tokens - low_prob_tokens
                    
                    ref_normalized_probs.append(probs)
                    ref_total_tokens.append(total_tokens)
                    ref_low_prob_tokens.append(low_prob_tokens)
                    ref_high_prob_tokens.append(high_prob_tokens)
                    
                    print(f"Reference Samples {i+1}: Total tokens: {total_tokens}, Low prob: {low_prob_tokens} ({low_prob_tokens/total_tokens*100:.2f}%), High prob: {high_prob_tokens} ({high_prob_tokens/total_tokens*100:.2f}%)")
            else:
                ref_normalized_probs.append([])
                ref_total_tokens.append(0)
                ref_low_prob_tokens.append(0)
                ref_high_prob_tokens.append(0)
                print(f"Reference Samples {i+1}: Empty sequence")


        all_total_tokens_tensor = torch.tensor(all_total_tokens, device=responses.device)
        all_low_prob_tokens_tensor = torch.tensor(all_low_prob_tokens, device=responses.device)
        all_high_prob_tokens_tensor = torch.tensor(all_high_prob_tokens, device=responses.device)

        ref_total_tokens_tensor = torch.tensor(ref_total_tokens, device=responses.device)
        ref_low_prob_tokens_tensor = torch.tensor(ref_low_prob_tokens, device=responses.device)
        ref_high_prob_tokens_tensor = torch.tensor(ref_high_prob_tokens, device=responses.device)

        all_low_prob_tokens_tensor_mean = all_low_prob_tokens_tensor.float().mean().item()
        all_high_prob_tokens_tensor_mean = all_high_prob_tokens_tensor.float().mean().item()
        ref_low_prob_tokens_tensor_mean = ref_low_prob_tokens_tensor.float().mean().item()
        ref_high_prob_tokens_tensor_mean = ref_high_prob_tokens_tensor.float().mean().item()


        all_low_prob_ratio_tensor = torch.zeros_like(all_total_tokens_tensor, dtype=torch.float32)
        all_high_prob_ratio_tensor = torch.zeros_like(all_total_tokens_tensor, dtype=torch.float32)

        ref_low_prob_ratio_tensor = torch.zeros_like(ref_total_tokens_tensor, dtype=torch.float32)
        ref_high_prob_ratio_tensor = torch.zeros_like(ref_total_tokens_tensor, dtype=torch.float32)

        valid_indices = all_total_tokens_tensor > 0
        if valid_indices.any():
            all_low_prob_ratio_tensor[valid_indices] = all_low_prob_tokens_tensor[valid_indices].float() / all_total_tokens_tensor[valid_indices].float()
            all_high_prob_ratio_tensor[valid_indices] = all_high_prob_tokens_tensor[valid_indices].float() / all_total_tokens_tensor[valid_indices].float()

        ref_valid_indices = ref_total_tokens_tensor > 0
        if ref_valid_indices.any():
            ref_low_prob_ratio_tensor[ref_valid_indices] = ref_low_prob_tokens_tensor[ref_valid_indices].float() / ref_total_tokens_tensor[ref_valid_indices].float()
            ref_high_prob_ratio_tensor[ref_valid_indices] = ref_high_prob_tokens_tensor[ref_valid_indices].float() / ref_total_tokens_tensor[ref_valid_indices].float()


        high_to_ref_high_ratio = torch.zeros_like(all_total_tokens_tensor, dtype=torch.float32)  # [batch_size]
        low_to_ref_low_ratio = torch.zeros_like(all_total_tokens_tensor, dtype=torch.float32)    # [batch_size]

        valid_for_ratio = (valid_indices & ref_valid_indices & (ref_high_prob_tokens_tensor > 0) & (ref_low_prob_tokens_tensor > 0))

        if valid_for_ratio.any():
            high_to_ref_high_ratio[valid_for_ratio] = all_high_prob_tokens_tensor[valid_for_ratio].float() / ref_high_prob_tokens_tensor[valid_for_ratio].float()
            
            low_to_ref_low_ratio[valid_for_ratio] = all_low_prob_tokens_tensor[valid_for_ratio].float() / ref_low_prob_tokens_tensor[valid_for_ratio].float()

        avg_high_to_ref_high = high_to_ref_high_ratio[valid_for_ratio].mean().item() if valid_for_ratio.any() else 0
        avg_low_to_ref_low = low_to_ref_low_ratio[valid_for_ratio].mean().item() if valid_for_ratio.any() else 0

        
        print("\n===== Relative ratio analysis =====")
        print(f"High probability token ratio (current/ref): {avg_high_to_ref_high:.4f}")
        print(f"Low probability token ratio (current/ref): {avg_low_to_ref_low:.4f}")

        print("\n===== Summary statistics =====")
        print("Original responses:")
        avg_low_prob_ratio = all_low_prob_ratio_tensor.mean().item()
        avg_high_prob_ratio = all_high_prob_ratio_tensor.mean().item()
        print(f"- Average low probability token ratio: {avg_low_prob_ratio:.4f} ({avg_low_prob_ratio*100:.2f}%)")
        print(f"- Average high probability token ratio: {avg_high_prob_ratio:.4f} ({avg_high_prob_ratio*100:.2f}%)")
        print(f"- Average total tokens: {all_total_tokens_tensor.float().mean().item():.1f}")

        print("\nReference responses:")
        ref_avg_low_prob_ratio = ref_low_prob_ratio_tensor.mean().item()
        ref_avg_high_prob_ratio = ref_high_prob_ratio_tensor.mean().item()
        print(f"- Average low probability token ratio: {ref_avg_low_prob_ratio:.4f} ({ref_avg_low_prob_ratio*100:.2f}%)")
        print(f"- Average high probability token ratio: {ref_avg_high_prob_ratio:.4f} ({ref_avg_high_prob_ratio*100:.2f}%)")
        print(f"- Average total tokens: {ref_total_tokens_tensor.float().mean().item():.1f}")

        
        if 'metrics' not in data.meta_info:
            data.meta_info['metrics'] = {}

        data.meta_info['metrics'].update({
            'token_stats/original_low_prob_ratio': avg_low_prob_ratio,
            'token_stats/original_high_prob_ratio': avg_high_prob_ratio,
            'token_stats/original_avg_tokens_mean': all_total_tokens_tensor.float().mean().item(),
            'token_stats/ref_low_prob_ratio': ref_avg_low_prob_ratio,
            'token_stats/ref_high_prob_ratio': ref_avg_high_prob_ratio,
            'token_stats/ref_avg_tokens_mean': ref_total_tokens_tensor.float().mean().item(),
            'token_stats/high_to_ref_high_ratio': avg_high_to_ref_high,
            'token_stats/low_to_ref_low_ratio': avg_low_to_ref_low,
            'token_stats/original_low_prob_tokens_mean': all_low_prob_tokens_tensor_mean,
            'token_stats/original_high_prob_tokens_mean': all_high_prob_tokens_tensor_mean,
            'token_stats/ref_low_prob_tokens_mean': ref_low_prob_tokens_tensor_mean,
            'token_stats/ref_high_prob_tokens_mean': ref_high_prob_tokens_tensor_mean
        })
        
        
        token_scores = data.batch['token_level_scores']       # [batch_size, seq_len]
        
        has_correct_prediction = (token_scores == 1).any(dim=1)  # [batch_size]
        correct_samples_mask = has_correct_prediction
        wrong_samples_mask = ~has_correct_prediction

        correct_samples_count = correct_samples_mask.sum().item()
        wrong_samples_count = wrong_samples_mask.sum().item()
        total_samples = len(responses)

        print(f"\n===== Sample statistics =====")
        print(f"Number of positive samples: {correct_samples_count}/{total_samples} ({correct_samples_count/total_samples*100:.2f}%)")
        print(f"Number of negative samples: {wrong_samples_count}/{total_samples} ({wrong_samples_count/total_samples*100:.2f}%)")
        
        if valid_indices.any():
            correct_indices = valid_indices & correct_samples_mask
            if correct_indices.any():
                correct_high_tokens = all_high_prob_tokens_tensor[correct_indices]
                correct_high_tokens_mean = correct_high_tokens.float().mean().item()
                
                correct_low_tokens = all_low_prob_tokens_tensor[correct_indices]
                correct_low_tokens_mean = correct_low_tokens.float().mean().item()
                
                correct_total_tokens = all_total_tokens_tensor[correct_indices]
                correct_high_ratio = (correct_high_tokens.float() / correct_total_tokens.float()).mean().item()
                correct_low_ratio = (correct_low_tokens.float() / correct_total_tokens.float()).mean().item()
            else:
                correct_high_tokens_mean = 0
                correct_low_tokens_mean = 0
                correct_high_ratio = 0
                correct_low_ratio = 0
            
            wrong_indices = valid_indices & wrong_samples_mask
            if wrong_indices.any():
                wrong_high_tokens = all_high_prob_tokens_tensor[wrong_indices]
                wrong_high_tokens_mean = wrong_high_tokens.float().mean().item()
                
                wrong_low_tokens = all_low_prob_tokens_tensor[wrong_indices]
                wrong_low_tokens_mean = wrong_low_tokens.float().mean().item()
                
                wrong_total_tokens = all_total_tokens_tensor[wrong_indices]
                wrong_high_ratio = (wrong_high_tokens.float() / wrong_total_tokens.float()).mean().item()
                wrong_low_ratio = (wrong_low_tokens.float() / wrong_total_tokens.float()).mean().item()
            else:
                wrong_high_tokens_mean = 0
                wrong_low_tokens_mean = 0
                wrong_high_ratio = 0
                wrong_low_ratio = 0
        else:
            correct_high_tokens_mean = 0
            correct_low_tokens_mean = 0
            correct_high_ratio = 0
            correct_low_ratio = 0
            wrong_high_tokens_mean = 0
            wrong_low_tokens_mean = 0
            wrong_high_ratio = 0
            wrong_low_ratio = 0
        
        correct_cos_values = []
        wrong_cos_values = []

        for i in range(token_scores.shape[0]):
            correct_positions = (token_scores[i] == 1).nonzero(as_tuple=True)[0]
            
            if len(correct_positions) > 0:
                # ratio_term_token[i, correct_positions] = torch.clamp(ratio_term[i] * lambda_c, min=0.0, max=5.0)

                cos_value = lambda_c * torch.cos(torch.clamp(low_to_ref_low_ratio[i], min=0.0, max=math.pi / 2))  # range [0,1]
                ratio_term_token[i, correct_positions] = cos_value
                correct_cos_values.append(cos_value.item() + 1)
            else:
                valid_positions = response_mask[i].nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    last_valid_pos = valid_positions[-1]
                    if initial_slope is not None:
                        cos_value = lambda_w_n * (torch.cos(torch.clamp(low_to_ref_low_ratio[i], min=0.0, max=math.pi / 2)) - 1) + dynamic_lambda_w_e * high_to_ref_high_ratio[i] - lambda_w_e
                    else:
                        cos_value = lambda_w_n * (torch.cos(torch.clamp(low_to_ref_low_ratio[i], min=0.0, max=math.pi / 2)) - 1) + (lambda_w_e / (math.pi / 2)) * high_to_ref_high_ratio[i] - lambda_w_e
                    ratio_term_token[i, last_valid_pos] = torch.clamp(cos_value, max=0.0)
                    wrong_cos_values.append(cos_value.item())

        ratio_term_token = ratio_term_token * response_mask

        token_level_rewards = token_level_rewards + ratio_term_token
        
        correct_cos_values = torch.tensor(correct_cos_values) if correct_cos_values else torch.tensor([])
        wrong_cos_values = torch.tensor(wrong_cos_values) if wrong_cos_values else torch.tensor([])

        correct_cos_mean = correct_cos_values.mean().item() if len(correct_cos_values) > 0 else 0
        correct_cos_min = correct_cos_values.min().item() if len(correct_cos_values) > 0 else 0
        correct_cos_max = correct_cos_values.max().item() if len(correct_cos_values) > 0 else 0

        wrong_cos_mean = wrong_cos_values.mean().item() if len(wrong_cos_values) > 0 else 0
        wrong_cos_min = wrong_cos_values.min().item() if len(wrong_cos_values) > 0 else 0
        wrong_cos_max = wrong_cos_values.max().item() if len(wrong_cos_values) > 0 else 0
        
        data.meta_info['metrics'].update({
            'samples/correct_count': correct_samples_count,
            'samples/wrong_count': wrong_samples_count,
            'samples/correct_ratio': correct_samples_count / total_samples,
            
            'cos_value/correct_mean': correct_cos_mean,
            'cos_value/correct_min': correct_cos_min,
            'cos_value/correct_max': correct_cos_max,
            
            'cos_value/wrong_mean': wrong_cos_mean,
            'cos_value/wrong_min': wrong_cos_min,
            'cos_value/wrong_max': wrong_cos_max,
            
            'token_stats/correct_high_tokens_mean': correct_high_tokens_mean,
            'token_stats/correct_low_tokens_mean': correct_low_tokens_mean,
            'token_stats/correct_high_ratio': correct_high_ratio,
            'token_stats/correct_low_ratio': correct_low_ratio,
            
            'token_stats/wrong_high_tokens_mean': wrong_high_tokens_mean,
            'token_stats/wrong_low_tokens_mean': wrong_low_tokens_mean,
            'token_stats/wrong_high_ratio': wrong_high_ratio,
            'token_stats/wrong_low_ratio': wrong_low_ratio,
        })

        
        
        if ppo:
            values = data.batch['values']
            responses = data.batch['responses']
            response_length = responses.size(-1)
            attention_mask = data.batch['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            token_level_rewards = data.batch['token_level_rewards']
        
        
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=response_mask,
            gamma=gamma,
            lam=lam
        )

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    else:
        raise NotImplementedError
    return data


# Update RayPPOTrainer with specific functions
class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator in [AdvantageEstimator.GAE, AdvantageEstimator.BINGO]:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

        # add attributes for tracking validation scores
        self.validation_scores = []
        self.validation_steps = []
        self.dynamic_lambda_w_e = self.config.algorithm.lambda_w_e
        self.initial_slope = None
        self.slope_scale = None
        self.low_slope_count = 0

        self.training_scores = []
        self.training_steps = []


    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        
        sample_data_sources = []

        # initialize a dictionary to track response lengths
        data_source_stats = {}

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            
            responses = test_batch.batch['responses']
            attention_mask = test_batch.batch['attention_mask']
            response_length = responses.size(1)
            response_mask = attention_mask[:, -response_length:]
            current_lengths = response_mask.sum(dim=1).cpu().numpy()  # [batch_size]
            
            data_sources = test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0])
            is_correct = (reward_tensor.sum(-1) > 0).cpu().numpy()
            sample_data_sources.extend(data_sources)

            for i, (response, length, is_correct_sample, data_source) in enumerate(
                    zip(responses, current_lengths, is_correct, data_sources)):
                
                if length > 0:
                    valid_tokens = response[:length]
                    text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                    
                    try:
                        result = compress_prompt(text, 0.6)
                        
                        sample_original_probs = result["original_probs"]
                        sample_normalized_probs = normalize_probs(sample_original_probs)
                        
                        total_tokens = 0
                        low_prob_tokens = 0
                        high_prob_tokens = 0
                        
                        for probs in sample_normalized_probs:
                            tokens_in_chunk = len(probs)
                            total_tokens += tokens_in_chunk
                            low_tokens_in_chunk = len([p for p in probs if p < self.config.algorithm.i_threshold])
                            low_prob_tokens += low_tokens_in_chunk
                            high_prob_tokens += tokens_in_chunk - low_tokens_in_chunk
                        
                        if data_source not in data_source_stats:
                            data_source_stats[data_source] = {
                                'total_count': 0,
                                'total_length': 0,
                                'total_high_prob': 0,
                                'total_low_prob': 0,
                                'correct_count': 0,
                                'correct_length': 0,
                                'correct_high_prob': 0,
                                'correct_low_prob': 0,
                                'wrong_count': 0,
                                'wrong_length': 0,
                                'wrong_high_prob': 0,
                                'wrong_low_prob': 0,
                            }
                        
                        data_source_stats[data_source]['total_count'] += 1
                        data_source_stats[data_source]['total_length'] += total_tokens
                        data_source_stats[data_source]['total_high_prob'] += high_prob_tokens
                        data_source_stats[data_source]['total_low_prob'] += low_prob_tokens
                        
                        if is_correct_sample:
                            data_source_stats[data_source]['correct_count'] += 1
                            data_source_stats[data_source]['correct_length'] += total_tokens
                            data_source_stats[data_source]['correct_high_prob'] += high_prob_tokens
                            data_source_stats[data_source]['correct_low_prob'] += low_prob_tokens
                        else:
                            data_source_stats[data_source]['wrong_count'] += 1
                            data_source_stats[data_source]['wrong_length'] += total_tokens
                            data_source_stats[data_source]['wrong_high_prob'] += high_prob_tokens
                            data_source_stats[data_source]['wrong_low_prob'] += low_prob_tokens
                    
                    except Exception as e:
                        print(f"error: {e}")
            
            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'acc/{data_source}'] = np.mean(rewards)
        
        for data_source, stats in data_source_stats.items():
            avg_total_length = stats['total_length'] / max(1, stats['total_count'])
            avg_total_high_prob = stats['total_high_prob'] / max(1, stats['total_count'])
            avg_total_low_prob = stats['total_low_prob'] / max(1, stats['total_count'])
            
            avg_correct_length = stats['correct_length'] / max(1, stats['correct_count'])
            avg_correct_high_prob = stats['correct_high_prob'] / max(1, stats['correct_count'])
            avg_correct_low_prob = stats['correct_low_prob'] / max(1, stats['correct_count'])
            
            avg_wrong_length = stats['wrong_length'] / max(1, stats['wrong_count']) 
            avg_wrong_high_prob = stats['wrong_high_prob'] / max(1, stats['wrong_count'])
            avg_wrong_low_prob = stats['wrong_low_prob'] / max(1, stats['wrong_count'])
            
            prefix = f'length/{data_source}'
            metric_dict[f'{prefix}/avg_length'] = avg_total_length
            metric_dict[f'{prefix}/avg_high_prob'] = avg_total_high_prob
            metric_dict[f'{prefix}/avg_low_prob'] = avg_total_low_prob
            
            metric_dict[f'{prefix}/correct_avg_length'] = avg_correct_length
            metric_dict[f'{prefix}/correct_avg_high_prob'] = avg_correct_high_prob
            metric_dict[f'{prefix}/correct_avg_low_prob'] = avg_correct_low_prob
            
            metric_dict[f'{prefix}/wrong_avg_length'] = avg_wrong_length
            metric_dict[f'{prefix}/wrong_avg_high_prob'] = avg_wrong_high_prob
            metric_dict[f'{prefix}/wrong_avg_low_prob'] = avg_wrong_low_prob
            
            metric_dict[f'{prefix}/total_samples'] = stats['total_count']
            metric_dict[f'{prefix}/correct_samples'] = stats['correct_count'] 
            metric_dict[f'{prefix}/wrong_samples'] = stats['wrong_count']
            
            correct_ratio = stats['correct_count'] / max(1, stats['total_count'])
            metric_dict[f'{prefix}/correct_ratio'] = correct_ratio
        
        self._log_validation_sample(sample_inputs, sample_outputs, sample_scores, self.global_steps, sample_data_sources)
        
        return metric_dict
        
    def _log_validation_sample(self, sample_inputs, sample_outputs, sample_scores, global_steps, data_sources=None):
        
        output_dir = os.path.join(self.config.trainer.default_local_dir, 'val')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        if data_sources is None:
            data_sources = ['unknown'] * len(sample_inputs)
        
        data_source_samples = {}
        for input_text, output_text, score, data_source in zip(sample_inputs, sample_outputs, sample_scores, data_sources):
            if data_source not in data_source_samples:
                data_source_samples[data_source] = []
            data_source_samples[data_source].append({
                'input': input_text,
                'output': output_text,
                'score': score
            })
        
        for data_source, samples in data_source_samples.items():
            safe_data_source = ''.join(c if c.isalnum() else '_' for c in data_source)
            output_path = os.path.join(output_dir, f'val_sample_{safe_data_source}_global_step_{global_steps}.jsonl')
            
            with open(output_path, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample))
                    f.write('\n')
            
            print(f"save {len(samples)} samples to {output_path}")
        
        all_output_path = os.path.join(output_dir, f'val_sample_global_step_{global_steps}.jsonl')
        with open(all_output_path, 'w') as f:
            for input_text, output_text, score in zip(sample_inputs, sample_outputs, sample_scores):
                f.write(json.dumps({'input': input_text, 'output': output_text, 'score': score}))
                f.write('\n')

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):     
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']


                        if self.config.algorithm.adv_estimator == AdvantageEstimator.BINGO:
                            if self.use_reference_policy:
                                print("====================generate ref response=====================")
                                batch_ref = DataProto.from_single_dict(batch_dict)

                                ref_gen_batch = batch_ref.pop(
                                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                                    non_tensor_batch_keys=['raw_prompt_ids']
                                )
                                ref_gen_output = self.ref_policy_wg.generate_sequences(ref_gen_batch)
                                batch_ref.batch['ref_responses'] = ref_gen_output.batch['responses']
                                batch_ref.batch['ref_attention_mask'] = ref_gen_output.batch['attention_mask']
                                batch_ref = batch_ref.union(ref_gen_output)

                                with _timer('ref_old_log_prob', timing_raw):
                                    ref_old_log_prob = self.ref_policy_wg.compute_log_prob(batch_ref)
                                    batch_ref = batch_ref.union(ref_old_log_prob)

                                ref_values = self.critic_wg.compute_values(batch_ref) if self.use_critic else None
                                if ref_values is not None:
                                    batch_ref = batch_ref.union(ref_values)

                                else:
                                    print("Warning: no critic found. Can't compute ref_advantages. Will skip.")
                            else:
                                print("Warning: adv_estimator=CUSTOM_FORMULA but no reference policy found.")
                            
                            batch.batch['ref_responses'] = batch_ref.batch['ref_responses']
                            batch.batch['ref_attention_mask'] = batch_ref.batch['ref_attention_mask']
                            batch.batch['ref_old_log_probs'] = batch_ref.batch['old_log_probs']

                        batch = compute_advantage(batch,
                                                adv_estimator=self.config.algorithm.adv_estimator,
                                                gamma=self.config.algorithm.gamma,
                                                lam=self.config.algorithm.lam,
                                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                lambda_c=self.config.algorithm.lambda_c,
                                                lambda_w_n=self.config.algorithm.lambda_w_n,
                                                lambda_w_e=self.config.algorithm.lambda_w_e,
                                                tokenizer=self.tokenizer,
                                                i_threshold=self.config.algorithm.i_threshold,
                                                initial_slope=self.initial_slope,
                                                dynamic_lambda_w_e=self.dynamic_lambda_w_e,
                                                ppo=self.config.algorithm.ppo)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                    
                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                if 'metrics' in batch.meta_info:
                    metrics.update(batch.meta_info['metrics'])
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                print("Available metrics keys:", list(metrics.keys()))
                training_score = metrics.get('critic/score/mean', None)
                if training_score is not None:
                    if not hasattr(self, 'training_scores'):
                        self.training_scores = []
                        self.training_steps = []
                    
                    self.training_scores.append(training_score)
                    self.training_steps.append(self.global_steps)
                    
                    if self.global_steps >= self.config.algorithm.slope_start_epoch and len(self.training_scores) >= 2:
                        look_back = min(self.config.algorithm.slope_period, len(self.training_scores) - 1)
                        x = self.training_steps[-look_back:]
                        y = self.training_scores[-look_back:]
                        
                        X = np.array(x).reshape(-1, 1)
                        Y = np.array(y)
                        
                        model = LinearRegression().fit(X, Y)
                        slope = model.coef_[0]
                        
                        if self.initial_slope is None:
                            self.initial_slope = slope
                            if self.initial_slope == 0:
                                self.slope_scale = self.config.algorithm.lambda_w_e / (math.pi / 2)
                            else:
                                self.slope_scale = (self.config.algorithm.lambda_w_e / (math.pi / 2)) / max(1e-6, abs(self.initial_slope))
                            print(f"\n===== original slope =====")
                            print(f"original slope: {self.initial_slope:.8f}")
                            print(f"scale: {self.slope_scale:.8f}")
                        
                        scaled_slope = slope * self.slope_scale
                        
                        print(f"\n===== slope scale =====")
                        print(f"original slope: {slope:.8f}")
                        print(f"scaled slope: {scaled_slope:.4f} (original slope mapped to {self.config.algorithm.lambda_w_e/(math.pi/2):.2f})")
                        print(f"look back {look_back} steps")
                        
                        if scaled_slope <= self.config.algorithm.slope_threshold:
                            self.low_slope_count += 1
                            
                            negative_lambda = -0.5 * self.low_slope_count
                            self.dynamic_lambda_w_e = max(-2 * self.config.algorithm.lambda_w_e / (math.pi / 2), negative_lambda)
                            
                            print(f"scaled slope <= {self.config.algorithm.slope_threshold}, continuous low slope count: {self.low_slope_count}")
                            print(f"lambda_w_e decrease to: {self.dynamic_lambda_w_e:.4f}")
                        else:
                            self.low_slope_count = 0
                            self.dynamic_lambda_w_e = scaled_slope
                            print(f"lambda_w_e: {self.dynamic_lambda_w_e:.4f}")
                        
                        metrics['lambda_w_e/train_raw_slope'] = slope
                        metrics['lambda_w_e/train_scaled_slope'] = scaled_slope
                        metrics['lambda_w_e/train_value'] = self.dynamic_lambda_w_e
                        metrics['lambda_w_e/train_low_slope_count'] = self.low_slope_count
                
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    return

                self.global_steps += 1
