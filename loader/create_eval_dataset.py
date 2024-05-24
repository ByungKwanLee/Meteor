import os
import json
import math
import glob
from config import *
from PIL import Image
import pandas as pd
import pyarrow.parquet as pq 
import torch.nn.functional as F
from eval.utils import *
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

class CreateEvalDataset(Dataset):
    def __init__(self):
        super(CreateEvalDataset, self).__init__()

        """
        Eval Datasets

        - VQAv2
        - GQA
        - SQA-IMG
        - VizWiz
        - TextVQA
        - POPE
        - MME
        - MMBench
        - MMBench-CN
        - QBench
        - MM-Vet
        - MMMU
        - MathVista
        - AI2D
        - HallusionBench
        - ChartQA
        - SEED
        - LLaVA Wild
        - BLINK
        - MathVerse

        """

        # dataset root path
        self.dataset_root_path = DATASET_ROOT

        # load test data
        pre_vqav2 = json.load(open(os.path.join(DATASET_ROOT, VQAV2)))
        pre_gqa = json.load(open(os.path.join(DATASET_ROOT, GQA)))
        pre_sqa = json.load(open(os.path.join(DATASET_ROOT, SQA)))
        pre_sqa_split = json.load(open(os.path.join(DATASET_ROOT, SQA_SPLIT)))
        pre_vizwiz = json.load(open(os.path.join(DATASET_ROOT, VIZWIZ)))
        pre_textvqa = json.load(open(os.path.join(DATASET_ROOT, TEXTVQA)))
        pre_textvqa_annotations = json.load(open(os.path.join(DATASET_ROOT, TEXTVQA_ANNOTATIONS)))
        pre_pope_popular = pd.read_json(os.path.join(DATASET_ROOT, POPE_POPULAR), lines=True)
        pre_pope_adversarial= pd.read_json(os.path.join(DATASET_ROOT, POPE_ADVERSARIAL), lines=True)
        pre_pope_random = pd.read_json(os.path.join(DATASET_ROOT, POPE_RANDOM), lines=True)
        pre_mme = json.load(open(os.path.join(DATASET_ROOT, MME)))
        pre_mmbench = pd.read_table(os.path.join(DATASET_ROOT, MMBENCH))
        pre_mmbench_dev = pd.read_table(os.path.join(DATASET_ROOT, MMBENCH_DEV))
        pre_mmbench_cn = pd.read_table(os.path.join(DATASET_ROOT, MMBENCH_CN))
        pre_mmbench_cn_dev = pd.read_table(os.path.join(DATASET_ROOT, MMBENCH_CN_DEV))
        pre_qbench = json.load(open(os.path.join(DATASET_ROOT, QBENCH)))
        pre_qbench_cn = json.load(open(os.path.join(DATASET_ROOT, QBENCH_CN)))
        pre_mmvet = json.load(open(os.path.join(DATASET_ROOT, MMVET)))
        mmmu_files = glob.glob(os.path.join(DATASET_ROOT, MMMU))
        pre_mmmu = [pq.read_pandas(os.path.join(DATASET_ROOT, mf)).to_pandas() for mf in mmmu_files]
        pre_mathvista1 = pq.read_pandas(os.path.join(DATASET_ROOT, MATHVISTA)).to_pandas()
        pre_ai2d = json.load(open(os.path.join(DATASET_ROOT, AI2D)))
        pre_hallusionbench = json.load(open(os.path.join(DATASET_ROOT, HALLUSIONBENCH)))
        pre_chartqa = json.load(open(os.path.join(DATASET_ROOT, CHARTQA)))
        pre_seed = json.load(open(os.path.join(DATASET_ROOT, SEED)))
        pre_llava = pd.read_json(os.path.join(DATASET_ROOT, LLAVA), lines=True)
        # pre_blink = json.load(open(os.path.join(DATASET_ROOT, BLINK)))
        pre_mathverse = json.load(open(os.path.join(DATASET_ROOT, MATHVERSE)))
        pre_mathverse_text_only = json.load(open(os.path.join(DATASET_ROOT, MATHVERSE_TEXT_ONLY)))
        pre_mmstar = pq.read_pandas(os.path.join(DATASET_ROOT, MMSTAR)).to_pandas()
        
        # data filtering
        vqav2 = self.vqav2_filtering(pre_vqav2)
        gqa = self.gqa_filtering(pre_gqa)
        sqa = self.sqa_filtering(pre_sqa, pre_sqa_split)
        vizwiz = self.vizwiz_filtering(pre_vizwiz)
        textvqa = self.textvqa_filtering(pre_textvqa, pre_textvqa_annotations)
        pope = self.pope_filtering([pre_pope_popular, pre_pope_adversarial, pre_pope_random])
        mme = self.mme_filtering(pre_mme)
        mmbench = self.mmbench_filtering(pre_mmbench)
        mmbench_dev = self.mmbench_filtering(pre_mmbench_dev)
        mmbench_cn = self.mmbench_filtering(pre_mmbench_cn)
        mmbench_cn_dev = self.mmbench_filtering(pre_mmbench_cn_dev)
        qbench = self.qbench_filtering(pre_qbench)
        qbench_cn = self.qbench_filtering(pre_qbench_cn)
        mmvet = self.mmvet_filtering(pre_mmvet)
        mmmu = self.mmmu_filtering(pre_mmmu)
        mathvista = self.mathvista_filtering(pre_mathvista1)
        ai2d = self.ai2d_filtering(pre_ai2d)
        hallusionbench = self.hallusionbench_filtering(pre_hallusionbench)
        chartqa = self.chartqa_filtering(pre_chartqa)
        seed = self.seed_filtering(pre_seed)
        llava = self.llava_filtering(pre_llava)
        # blink = self.blink_filtering(pre_blink)
        mathverse = self.mathverse_filtering(pre_mathverse, pre_mathverse_text_only)
        mmstar = self.mmstar_filtering(pre_mmstar)
        
        # merging
        self.data = {
            'vqav2': vqav2,
            'gqa': gqa,
            'sqa':sqa,
            'vizwiz': vizwiz,
            'textvqa': textvqa,
            'pope': pope,
            'mme': mme,
            'mmbench': mmbench,
            'mmbench_dev': mmbench_dev,
            'mmbench_cn': mmbench_cn,
            'mmbench_cn_dev': mmbench_cn_dev,
            'qbench': qbench,
            'mm-vet': mmvet,
            'mmmu': mmmu,
            'mathvista': mathvista,
            'ai2d': ai2d,
            'hallusionbench': hallusionbench,
            'chartqa': chartqa,
            'seed': seed,
            'llava': llava,
            # 'blink': blink,
            'mathverse' : mathverse,
            'mmstar' : mmstar
        }        


    def vqav2_filtering(self, pre_data):
        data = []
        for x in pre_data['questions']:
            data.append({'image': f"VQAv2/test2015/COCO_test2015_{x['image_id']:012d}.jpg",
                        'question': x['question'],
                        'id': x['question_id']})
        return data
    
    def gqa_filtering(self, pre_data):
        data = []
        for qid, x in pre_data.items():
            data.append({'image': f"gqa/images/{x['imageId']}.jpg",
                        'question': x['question'],
                        'id': qid})
        return data
    
    def sqa_filtering(self, pre_data, pre_sqa_split):
        data = []
        questions = {idx: pre_data[idx] for idx in pre_sqa_split['test']}
        for qid, x in questions.items():
            if x['image'] is not None:
                choices = '\n'.join(f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(x['choices']))
                question = '\n'.join([x['hint'], x['question'], choices])
                data.append({'image': f"ScienceQA/images/test/{qid}/image.png",
                            'question': question,
                            'id': qid,
                            'candidates': x['choices'],
                            'gt': x['answer']})
        return data
    
    def vizwiz_filtering(self, pre_data):
        data = []
        for qid, x in enumerate(pre_data):
            data.append({'image': f"VizWiz/test/{x['image']}",
                        'question': x['question'],
                        'id': qid})
        return data
    
    def textvqa_filtering(self, pre_data, annotations):
        data = []
        for x, answer in zip(pre_data, annotations['data']):
            data.append({'image': f"TextVQA/train_images/{x['image']}",
                        'question': x['text'],
                        'id': x['question_id'],
                        'gt': answer['answers']})
        return data
    
    def pope_filtering(self, pre_data):
        data = []
        categories = ['adversarial', 'popular', 'random']
        for category, split in zip(categories, pre_data):
            for _, x in split.iterrows():
                data.append({'image': f"coco2014/val2014/{x['image']}",
                            'question': x['text'],
                            'id': x['question_id'], 
                            'category': category})
        return data
    
    def mme_filtering(self, pre_data):
        data = []
        for x in pre_data:
            data.append({'image': f"MME_Benchmark_release_version/{x['image']}",
                        'question': x['text'],
                        'id': x['question_id'],
                        'category': x['category']})
        return data
    
    def mmbench_filtering(self, pre_data):
        data = []
        for _, x in pre_data.iterrows():
            options = ['A', 'B', 'C', 'D']
            choice_list = [choice for choice in options if not self.is_none(x[choice])]
            choices = '\n'.join(f"{chr(ord('A') + i)}. {x[choice]}" for i, choice in enumerate(choice_list))
            question = '\n'.join([x['question'], choices])

            if not self.is_none(x['hint']):
                question = '\n'.join([x['hint'], question])
        
            data.append({'image': x['image'],
                        'question': question,
                        'id': x['index'],
                        'answer': x['answer'] if 'answer' in x else None})
        return data
    
    def qbench_filtering(self, pre_data):
        data = []
        for qid, x in enumerate(pre_data):
            choices = '\n'.join(f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(x['candidates']))
            question = '\n'.join([x['question'], choices])
            data.append({'image': f"LLVisionQA-QBench/images/{x['img_path']}",
                        'question': question,
                        'id': qid,
                        'candidates': x['candidates'],
                        'gt': x['correct_ans']})
        return data
    
    def mmvet_filtering(self, pre_data):
        data = []
        for qid, x in pre_data.items():
            data.append({'image': f"mm-vet/images/{x['imagename']}",
                        'question': x['question'],
                        'id': qid,
                        'gt': x['answer'],
                        'capability': x['capability']})
        return data
    
    def mmmu_filtering(self, pre_data):
        data = []
        for split in pre_data:
            for _, x in split.iterrows():
                index2ans, all_choices = self.get_multi_choice_info(eval(x['options']))
                choices = ' '.join([f"{k}. {v}" for k,v in index2ans.items()])
                question = '\n'.join([x['question'], choices])
                num_images = count_unique_image_tokens(question)
                data.append({'images': [x[f"image_{i+1}"]['bytes'] for i in range(num_images)],
                            'question': replace_image_tokens(question),
                            'id': x['id'],
                            'question_type': x['question_type'],
                            'gt': x['answer'],
                            'index2ans': index2ans,
                            'all_choices': all_choices})
        return data
    
    def mathvista_filtering(self, pre_data):
        data = []
        for _, x in pre_data.iterrows():
            skills = x['metadata']['skills'].tolist()
            x['metadata']['skills'] = skills
            choices = x['choices'].tolist() if x['choices'] is not None else None
            data.append({'image': f"MathVista/{x['image']}",
                        'question': x['query'],
                        'question_type': x['question_type'],
                        'answer': x['answer'],
                        'answer_type': x['answer_type'],
                        'choices': choices,
                        'metadata': x['metadata'],
                        'precision': x['precision'],
                        'id': x['pid']})
        return data
    
    def ai2d_filtering(self, pre_data):
        data = []
        for x in pre_data:
            choices = ' '.join(f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(x["metadata"]["answerTexts"]))
            question = '\n'.join([x['question'], choices])
            image = f"ai2d/abc_images/{x['imageName']}" if x['metadata']['abcLabel'] else f"ai2d/images/{x['imageName']}"
            data.append({'image': image,
                         'question': question,
                        'id': x['metadata']['questionId'],
                        'gt': x['metadata']['correctAnswer']})
        return data
    
    def hallusionbench_filtering(self, pre_data):
        data = []
        for qid, x in enumerate(pre_data):
            if x['filename'] is None:
                img_path = ""
                question = x['question']
            else:
                img_path = f"HallusionBench/hallusion_bench/{x['filename'][2:]}".format()
                question =  "<image>" + x['question']
            data.append({'image': img_path,
                        'question': question,
                        'id': qid,
                        'gt': x['gt_answer']})
        return data

    def chartqa_filtering(self, pre_data):
        data = []
        for qid, x in enumerate(pre_data):
            data.append({'image': f"chartqa/test/png/{x['imgname']}",
                        'question': x['query'],
                        'id': x['imgname'],
                        'gt': x['label']})
        return data
    
    def seed_filtering(self, pre_data):
        data = []
        for x in pre_data['questions']:
            if x['data_type'] != 'image':
                continue
            choice_list = [key for key in x.keys() if 'choice' in key]
            choices = '\n'.join(f"{chr(ord('A') + i)}. {x[choice]}" for i, choice in enumerate(choice_list))
            question = '\n'.join([x['question'], choices])
            data.append({'image': f"SEED-Bench/SEED-Bench-image/{x['data_id']}",
                        'question': question,
                        'id': x['question_id'],
                        'question_type': x['question_type_id'],
                        'gt': x['answer']})
        return data
    
    def llava_filtering(self, pre_data):
        data = []
        for _, x in pre_data.iterrows():
            data.append({'image': f"llava-bench-in-the-wild/images/{x['image']}",
                        'question': x['text'],
                        'id': x['question_id'],
                        "category": x['category']})
        return data
    
    def blink_filtering(self, pre_data):
        data = []
        # TODO
        return data
    
    def mathverse_filtering(self, pre_data, pre_data_text_only):
        data = []
        for x in pre_data:
            data.append({'image': f"MathVerse/images/{x['image']}",
                        'question': "<image>" + x['query_wo'],
                        # 'question': "<image>" + x['query_cot'],
                        'id': x['sample_index'],
                        'problem_index': x['problem_index'],
                        'problem_version': x['problem_version'],
                        'gt' : x['answer'],
                        'question_type': x['question_type'],
                        'metadata' : x['metadata'],
                        'query_cot' : x['query_cot'],
                        'origin_question': x['question']
                        })
        offset = len(pre_data)
        for x in pre_data_text_only:
            data.append({'image': "",
                        'question': x['query_wo'],
                        # 'question': x['query_cot'],
                        'id': str(int(x['sample_index']) + offset),
                        'problem_index': x['problem_index'],
                        'problem_version': x['problem_version'],
                        'gt' : x['answer'],
                        'question_type': x['question_type'],
                        'metadata' : x['metadata'],
                        'query_cot' : x['query_cot'],
                        'origin_question': x['question']
                        })
            
        return data
    
    def is_none(self, value):
        return type(value) is float and math.isnan(value)

    def get_options(self, row, options):
        parsed_options = []
        for option in options:
            option_value = row[option]
            if self.is_none(option_value):
                break
            parsed_options.append(option_value)
        return parsed_options

    def __len__(self):
        return len(self.data)
    
    def get_multi_choice_info(self, options):
        """
        Given the list of options for multiple choice question
        Return the index2ans and all_choices
        """
        
        start_chr = 'A'
        all_choices = []
        index2ans = {}
        for i, option in enumerate(options):
            index2ans[chr(ord(start_chr) + i)] = option
            all_choices.append(chr(ord(start_chr) + i))

        return index2ans, all_choices
    
    def mmstar_filtering(self, pre_data):
        data = []
        for _, x in pre_data.iterrows():
            data.append({'id' : x['index'],
                        'question': x['question'],
                        'answer': x['answer'],
                        'category': x['category'],
                        'l2_category': x['l2_category'],
                        # 'bench': x['bench'],
                        'image': x['image']})
        return data