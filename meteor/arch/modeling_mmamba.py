# Transformers
import re
import torch
from torch import nn
from utils.utils import *
from typing import Optional, Tuple, Union
from transformers import MambaForCausalLM
from transformers import LlavaNextForConditionalGeneration, LlavaForConditionalGeneration

class MambaCache:
    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = {
            i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }

# Dataclass & ModelOutput
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
@dataclass
class MambaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaCache] = None
    tor_features: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class MeteorMambaForCausalLM(MambaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        # initialize other projections for Vision and tor
        self.vision_proj = self.build_vision_projector(1024, self.config.hidden_size)
        self.tor_proj = self.build_vision_projector(self.config.hidden_size, 4096)
        
        # replacing embedding size of mamba with that of meteor
        self.backbone.embeddings = nn.Embedding(num_embeddings=92546,
                                                embedding_dim=self.config.hidden_size)

        # image processing variable
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,-1,1,1) * 255
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,-1,1,1) * 255

    def image_processor(self, images):
        norm_images = (images - self.mean.to(images.device)) / self.std.to(images.device)
        return norm_images

    @staticmethod
    def build_vision_projector(mm_hidden_size, hidden_size):
        projector_type = 'mlp2x_gelu'
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(mm_hidden_size, hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(hidden_size, hidden_size))
            return nn.Sequential(*modules)

        raise ValueError(f'Unknown projector type: {projector_type}')

    def eval_process(
        self,
        inputs,
        tokenizer,
        device,
        img_token_number,
    ):
        batched_image=[]
        batched_qa_prompt=[]
        for _input in inputs:

            # Visualization
            # imim = _input['image'].cpu().permute(1, 2, 0)

            # adding <image> to question if not included despite being an image, and adding system prompt and <tor> prompt 
            if 'image' in _input.keys() and not '<image>' in _input['question']: _input['question'] = '<image>\n' + _input['question']

            # make question, rationale, and answer
            question = make_instruction_for_mmamba(question=_input['question'])

            # add bundle image tokens if it has <image> token
            question = add_bundle_tokens(question, '<image>', img_token_number) 

            # making batched moai prompt
            if 'image' in _input.keys() and _input['image'] != None: batched_image.append(_input['image'].to(device))
            batched_qa_prompt.append(question)

        '''For Final Outputs'''
        qa_prompts = tokenizer(batched_qa_prompt, padding='longest', return_tensors="pt", add_special_tokens=False)

        # [1] input_ids
        input_ids = qa_prompts.input_ids.to(device)

        # image or only text?
        if len(batched_image):
            # [2] pixel values
            try:
                pixel_values = self.image_processor(torch.stack(batched_image)).to(device)
                assert pixel_values.dim() == 4
            except:
                new_batched_image = []
                for batched_image_element in batched_image:
                    if batched_image_element.dim() == 3:
                        new_batched_image.append(batched_image_element.unsqueeze(0))
                    else:
                        new_batched_image.append(batched_image_element)
                pixel_values = self.image_processor(torch.cat(new_batched_image, dim=0)).to(device)

            return {"input_ids": input_ids, "image": pixel_values}
        else:
            return {"input_ids": input_ids}


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
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_features: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        # labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if image_features is not None and input_ids.shape[1] != 1:
                image_features = self.vision_proj(image_features)
                self._merge_input_embeds_with_image_features(image_features, inputs_embeds, input_ids)

        mamba_outputs = self.backbone(
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )
        hidden_states = mamba_outputs[0]

        # logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        # if labels is not None:
        #     # move labels to correct device to enable model parallelism
        #     labels = labels.to(logits.device)
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + mamba_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            cache_params=mamba_outputs.cache_params,
            tor_features=self.tor_proj(hidden_states[torch.where(input_ids==self.config.tor_token_index)]),
            hidden_states=mamba_outputs.hidden_states,
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, cache_params: Optional[MambaCache] = None, inputs_embeds=None, image_features=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds, "image_features":image_features}
        else:
            model_inputs = {"input_ids": input_ids, "image_features":image_features}

        model_inputs["cache_params"] = cache_params
        return model_inputs