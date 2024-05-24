# System
import torch
from torch import nn
from utils.utils import *
import torch.utils.checkpoint
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple, Union
from .build_module import build_vision_projector, build_vision_tower
from .modeling_internlm2 import InternLM2Model, InternLM2PreTrainedModel

# Dataclass & ModelOutput
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
@dataclass
class MeteorCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    tor_features: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class MeteorForCausalLM(InternLM2PreTrainedModel):
    _auto_class = 'AutoModelForCausalLM'

    _tied_weights_keys = ['output.weight']

    def __init__(self, config):
        super().__init__(config)
        
        # Model
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size-2, bias=False)
        self.max_length = config.max_length

        # Initialize weights and apply final processing
        self.post_init()

        # Vision Encoder
        self.vit = build_vision_tower()

        # Vision Projection
        self.vision_proj = build_vision_projector()

    def eval_process(
        self,
        inputs,
        data,
        tokenizer,
        device,
        img_token_number,
    ):
        
        batched_qa_prompt=[]
        for _input in inputs:

            # Visualization
            # imim = _input['image'].cpu().permute(1, 2, 0)

            # make question, rationale, and answer
            question = make_instruction_for_eval_meteor(_input['question'], data)

            # add bundle image tokens if it has <image> token
            question = add_bundle_tokens(question, '<image>', img_token_number) 

            batched_qa_prompt.append(question)

        '''For Final Outputs'''
        qa_prompts = tokenizer(batched_qa_prompt, padding='longest', return_tensors="pt", add_special_tokens=False)

        # [1] input_ids
        input_ids = qa_prompts.input_ids.to(device)
  
        # [2] attention_mask
        attention_mask = qa_prompts.attention_mask.to(device)

        # [3] im_mask
        im_mask = torch.zeros_like(input_ids).bool()
        im_mask[torch.where(input_ids==self.config.image_token_index)] = True

        return {"input_ids": input_ids, 
                "attention_mask": attention_mask, 
                "im_mask": im_mask,
                }

    def clip_features(self, image):
        self.vit.eval()
        return self.vit(image)

    def _merge_input_embeds_with_tor_features(self, tor_features, inputs_embeds, input_ids):
        
        # batch index for image feature
        batch_ind_tor_feature = 0

        for ind, input_id in enumerate(input_ids):
            matching = torch.where(input_id==self.config.tor_token_index)
            num_tor_tokens_per_one_sample = len(matching[0])
            inputs_embeds[ind][matching] = tor_features[batch_ind_tor_feature: batch_ind_tor_feature+num_tor_tokens_per_one_sample].to(inputs_embeds.dtype)
            batch_ind_tor_feature += num_tor_tokens_per_one_sample

    def _merge_input_embeds_with_image_features(self, image_features, inputs_embeds, input_ids):
        
        # batch index for image feature
        batch_ind_image_feature = 0

        # shape of image_features
        _, C, D = image_features.shape

        for ind, input_id in enumerate(input_ids):
            matching = torch.where(input_id==self.config.image_token_index)
            num_image_tokens_per_one_sample = len(matching[0]) // C
            inputs_embeds[ind][matching] = image_features[batch_ind_image_feature: batch_ind_image_feature+num_image_tokens_per_one_sample].view(-1, D)
            batch_ind_image_feature += num_image_tokens_per_one_sample

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_features: torch.FloatTensor = None,
        tor_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        im_mask: torch.BoolTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MeteorCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if image_features is not None and input_ids.shape[1] != 1:
                image_features = self.vision_proj(image_features.to(inputs_embeds.dtype))
                self._merge_input_embeds_with_image_features(image_features, inputs_embeds, input_ids)

            # 3. Merge text and <tor>
            if tor_features is not None and input_ids.shape[1] != 1:
                self._merge_input_embeds_with_tor_features(tor_features, inputs_embeds, input_ids)

            # In case input_ids.shape[1] == 1 & image_features==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and image_features is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                im_mask = torch.zeros(inputs_embeds.shape[:2]).bool().to(inputs_embeds.device)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            im_mask=im_mask,
        )

        hidden_states = outputs[0]
        logits = self.output(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MeteorCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            tor_features=hidden_states[torch.where(input_ids==self.config.tor_token_index)],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      image_features=None,
                                      tor_features=None, 
                                      im_mask=None,
                                      **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "image_features": image_features,
                "tor_features": tor_features,
                "im_mask": im_mask,
            }
        )
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past), )
        return reordered_past