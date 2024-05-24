    
# System
import os
import PIL
import json
from config import *
from PIL import Image
from utils.utils import *
import torch.nn.functional as F
PIL.Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor

class TrainMeteorDataset(Dataset):
    def __init__(self, resol):
        
        # resolution
        self.resol = resol
        self.data = json.load(open(os.path.join(DATASET_ROOT, METEOR_DATASET)))

    def __getitem__(self, index):

        if 'image' in self.data[index].keys():
            
            # img_path / instruction
            img_path = self.data[index]['image']
            conversations = self.data[index]['conversations']

            # img url -> img -> resized-img
            img_tensor = pil_to_tensor(Image.open(os.path.join(DATASET_ROOT, img_path)).convert("RGB"))
            resized_img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(self.resol, self.resol), mode='bicubic').squeeze(0)
            
            return {'id': self.data[index]['id'], 'image': resized_img_tensor, 'conversations': conversations}
        
        else:
            return {'id': self.data[index]['id'], 'conversations': self.data[index]['conversations']}
        
    def __len__(self):
        return len(self.data)

def main():

    # Meteor Dataset
    train_meteor_dataset = TrainMeteorDataset(resol=1280)
    train_meteor_dataloader = DataLoader(train_meteor_dataset, 
                                batch_size=1, 
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True,
                                collate_fn=lambda x: x)
    # print statistics
    print_data_statistics(train_meteor_dataset.data)
    
    # Checking Dataset for Debugging
    for inputs in train_meteor_dataloader:


        try:
            img = inputs[0]['image']
        except:
            pass
        name_id = inputs[0]['id']
        question = inputs[0]['conversations'][0]['value']
        rationale = inputs[0]['conversations'][1]['rationale']
        answer = inputs[0]['conversations'][1]['value']

        print(f'ID:\n\n{name_id}\n\n')
        print(f'Question:\n\n{question}\n\n')
        print(f'Rationale:\n\n{rationale}\n\n')
        print(f'Answer:\n\n{answer}')
        print('') # Debugging Point



if __name__ == "__main__": main()



