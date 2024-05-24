import os
import torch
import argparse
import base64
from config import *
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from utils.utils import *
from datetime import timedelta
import torch.nn.functional as F
from torch.utils.data import DataLoader
from meteor.load_mmamba import load_mmamba
from meteor.load_meteor import load_meteor
from eval.create_evaluator import Evaluator
from loader.create_eval_dataset import CreateEvalDataset
from accelerate import Accelerator, InitProcessGroupKwargs
from torchvision.transforms.functional import pil_to_tensor

class EvalDataset(CreateEvalDataset):
    def __init__(self, resol):
        super().__init__()

        # resolution
        self.resol = resol

        # select dataset
        self.eval_dataset = None

    def __getitem__(self, index):
        # img_path
        if 'image' in self.eval_dataset[index]:
            img_path = self.eval_dataset[index]['image']

            if img_path == "":
                # self.eval_dataset[index].update({'image': None})
                del self.eval_dataset[index]['image']
                return self.eval_dataset[index]
        else:
            # in case of multiple images like MMMU
            img_paths = self.eval_dataset[index]['images']
            images = [Image.open(BytesIO(img)).convert("RGB") for img in img_paths]
            img_tensors = [pil_to_tensor(img) for img in images]
            resized_img_tensors = [F.interpolate(img.unsqueeze(0), size=(self.resol, self.resol), mode='bicubic').squeeze(0) for img in img_tensors]
            concat_img = torch.stack(resized_img_tensors)
            self.eval_dataset[index].update({'image': concat_img})
            return self.eval_dataset[index]
        
        # img may contain encoded data
        try:
            image = Image.open(os.path.join(DATASET_ROOT, img_path)).convert("RGB")
        except:
            try: 
                # correct file names for hallusionbench
                if img_path.find('png') != -1:
                    new_img_path = img_path.replace('png', 'PNG')
                else:
                    new_img_path = img_path.replace('PNG', 'png')
                image = Image.open(os.path.join(DATASET_ROOT, new_img_path)).convert("RGB")
            except:
                try : 
                    image = Image.open(BytesIO(base64.b64decode(img_path))).convert("RGB")
                except :
                    image = Image.open(BytesIO(img_path)).convert("RGB")
        
        img_tensor = pil_to_tensor(image)
        resized_img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(self.resol, self.resol), mode='bicubic').squeeze(0)
        self.eval_dataset[index].update({'image': resized_img_tensor})

        return self.eval_dataset[index]

    def __len__(self):
        return len(self.eval_dataset)
    
    def update_dataset(self, dataset):
        self.eval_dataset = self.data[dataset]
    
def test(args):
    # Accelerator for DDP, FSDP, DeepSpeed, etc [Should First Call]
    accel = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))])

    # loading meteor model
    mmamba = load_mmamba('BK-Lee/Meteor-Mamba').cuda()
    meteor, tok_meteor = load_meteor('BK-Lee/Meteor-MLM', bits=4)

    # freeze model
    freeze_model(mmamba)
    freeze_model(meteor)

    # Select datasets to eval
    if args.dataset[0] == "all":
        eval_datasets = EVAL_DATASETS
    else:
        eval_datasets = args.dataset

    # Initialize dataset & evaluator
    test_dataset = EvalDataset(resol=args.resol)
    evaluator = Evaluator()
    results = {}

    for data in eval_datasets:
        # Update dataset & evaluator
        evaluator.reset()
        test_dataset.update_dataset(dataset=data)
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size=args.batch_size, 
                                    shuffle=False,
                                    num_workers=16,
                                    pin_memory=True,
                                    collate_fn=lambda x: x)

        # accel model
        mmamba = mmamba.cuda()
        
        # cpu -> gpu
        for param in meteor.parameters():
            if not param.is_cuda:
                param.data = param.to(accel.device)
        

        # Accel distributed
        test_dataloader = accel.prepare(test_dataloader)

        # progress bar
        prog_bar = tqdm(test_dataloader, disable=not accel.is_local_main_process, total=len(test_dataloader))


        # eval start
        for j, inputs in enumerate(prog_bar):

            # memory opt
            memory_optimization()

            # Generate
            with torch.inference_mode():

                # Meteor Mamba
                mmamba_inputs = mmamba.eval_process(inputs=inputs, tokenizer=tok_meteor, device=accel.device, img_token_number=args.img_token_num)
                if 'image' in mmamba_inputs.keys():
                    clip_features = meteor.clip_features(mmamba_inputs['image'])
                    mmamba_inputs.update({"image_features": clip_features})
                mmamba_outputs = mmamba(**mmamba_inputs)
                
                # Meteor
                meteor_inputs = meteor.eval_process(inputs=inputs, data=data, tokenizer=tok_meteor, device=accel.device, img_token_number=args.img_token_num)
                if 'image' in mmamba_inputs.keys():
                    meteor_inputs.update({"image_features": clip_features})
                meteor_inputs.update({"tor_features": mmamba_outputs.tor_features})
                generate_ids = meteor.generate(**meteor_inputs, do_sample=False, num_beams=3, max_new_tokens=get_max_new_tokens(data), use_cache=True)

            # # image visualization
            # # imim = inputs[0]['image'].cpu().permute(1,2,0).numpy()
            decoded_text = tok_meteor.batch_decode(generate_ids, skip_special_tokens=True)
            
            # save predictions
            all_predictions = [x.split("[UNUSED_TOKEN_146]assistant\n")[-1].split("[UNUSED_TOKEN_145]")[0].strip() for x in decoded_text]
            for x in inputs: 
                if 'image' in x:
                    del x['image']
            evaluator.process(inputs, all_predictions)

        # wait for everyone
        print(f"[Device: {accel.device}] Finished!")
        accel.wait_for_everyone()

        # memory opt
        memory_optimization()

        # evaluate on dataset
        results[data] = evaluator.evaluate('Meteor', data, accel)
    
    accel.print(results)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='eval', type=str)
    parser.add_argument('--dataset', 
                        default='mme', nargs='+', 
                        help='all|vqav2|gqa|sqa|vizwiz|textvqa|pope|mme|mmbench\nmmbench_cn|qbench|mm-vet|mmmu|mathvista|ai2d\nhallusionbench|chartqa|seed|llava|blink|mathverse')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--resol', default=490, type=int)
    parser.add_argument('--bits', default=4, type=int)
    args = parser.parse_args()

    # image token num
    args.img_token_num = (args.resol // 14) ** 2

    # test
    test(args)

