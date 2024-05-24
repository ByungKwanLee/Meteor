import time
import torch
from config import *
from PIL import Image
from utils.utils import *
import torch.nn.functional as F
from meteor.load_mmamba import load_mmamba
from meteor.load_meteor import load_meteor
from torchvision.transforms.functional import pil_to_tensor

# User prompt
prompt_type='with_image' # text_only / with_image
img_path='figures/demo.png'
question='Provide the detail of the image'

# loading meteor model
mmamba = load_mmamba('BK-Lee/Meteor-Mamba').cuda()
meteor, tok_meteor = load_meteor('BK-Lee/Meteor-MLM', bits=4)

# freeze model
freeze_model(mmamba)
freeze_model(meteor)

# Device
device = torch.cuda.current_device()

# prompt type -> input prompt
image_token_number = int((490/14)**2)
if prompt_type == 'with_image':
    # Image Load
    image = F.interpolate(pil_to_tensor(Image.open(img_path).convert("RGB")).unsqueeze(0), size=(490, 490), mode='bicubic').squeeze(0)
    inputs = [{'image': image, 'question': question}]
elif prompt_type=='text_only':
    inputs = [{'question': question}]

# Generate
with torch.inference_mode():

    # Meteor Mamba
    mmamba_inputs = mmamba.eval_process(inputs=inputs, tokenizer=tok_meteor, device=device, img_token_number=image_token_number)
    if 'image' in mmamba_inputs.keys():
        clip_features = meteor.clip_features(mmamba_inputs['image'])
        mmamba_inputs.update({"image_features": clip_features})
    mmamba_outputs = mmamba(**mmamba_inputs)
    
    # Meteor
    meteor_inputs = meteor.eval_process(inputs=inputs, data='demo', tokenizer=tok_meteor, device=device, img_token_number=image_token_number)
    if 'image' in mmamba_inputs.keys():
        meteor_inputs.update({"image_features": clip_features})
    meteor_inputs.update({"tor_features": mmamba_outputs.tor_features})

    # Generation
    generate_ids = meteor.generate(**meteor_inputs, do_sample=True, max_new_tokens=128, top_p=0.95, temperature=0.9, use_cache=True)

# Text decoding
decoded_text = tok_meteor.batch_decode(generate_ids, skip_special_tokens=True)[0].split('assistant\n')[-1].split('[U')[0].strip()
print(decoded_text)
