import torch
import gradio as gr
from config import *
from PIL import Image
from utils.utils import *
from threading import Thread
import torch.nn.functional as F
from meteor.load_mmamba import load_mmamba
from meteor.load_meteor import load_meteor
from transformers import TextIteratorStreamer
from torchvision.transforms.functional import pil_to_tensor

# loading meteor model
mmamba = load_mmamba('BK-Lee/Meteor-Mamba').cuda()
meteor, tok_meteor = load_meteor('BK-Lee/Meteor-MLM', bits=4)

# device
device = torch.cuda.current_device()

# freeze model
freeze_model(mmamba)
freeze_model(meteor)

# previous length
previous_length = 0


def threading_function(inputs, image_token_number, streamer):
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

    generation_kwargs = meteor_inputs
    generation_kwargs.update({'streamer': streamer})
    generation_kwargs.update({'do_sample': True})
    generation_kwargs.update({'max_new_tokens': 128})
    generation_kwargs.update({'top_p': 0.95})
    generation_kwargs.update({'temperature': 0.9})
    generation_kwargs.update({'use_cache': True})
    return meteor.generate(**generation_kwargs)

def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot_streaming(message, history):

    # prompt type -> input prompt
    image_token_number = int((490/14)**2)
    if len(message['files']) != 0:
        # Image Load
        image = F.interpolate(pil_to_tensor(Image.open(message['files'][0]).convert("RGB")).unsqueeze(0), size=(490, 490), mode='bicubic').squeeze(0)
        inputs = [{'image': image, 'question': message['text']}]
    else:
        inputs = [{'question': message['text']}]

    # [4] Meteor Generation
    with torch.inference_mode():
        # kwargs
        streamer = TextIteratorStreamer(tok_meteor, skip_special_tokens=True)

        # Threading generation
        thread = Thread(target=threading_function, kwargs=dict(inputs=inputs, image_token_number=image_token_number, streamer=streamer))
        thread.start()

        # generated text
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
        generated_text

    # Text decoding
    response = generated_text.split('assistant\n')[-1].split('[U')[0].strip()

    buffer = ""
    for character in response:
        buffer += character
        yield buffer

demo = gr.ChatInterface(fn=bot_streaming, title="Meteor", 
                        description="Meteor",
                        stop_btn="Stop Generation", multimodal=True)
demo.launch(debug=True)









