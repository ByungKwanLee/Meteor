import os
import gc
import math
import torch
import base64
import numpy as np
from config import *
import torch.nn.functional as F

def memory_optimization():
    # memory deallocation
    gc.collect()

    # removing cache
    torch.cuda.empty_cache()

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad=False

def find_special_token(string, special_token):
    start = 0
    while True:
        start = string.find(special_token, start)
        if start == -1: return
        yield start
        start += len(special_token) # use start += 1 to find overlapping matches

def insert_tor(sentence, tor_count):
    words = sentence.split()
    gap =  len(words) // (tor_count-1)

    # filtering
    if 0<=gap<=2:
        return False

    count = 0
    result = ""
    for i, word in enumerate(words):
        if 0<i<len(words)-1:
            result+=' '
        if i % gap == 0 and count != tor_count-1:
            result += '<tor>'
            count += 1
        result += word
    result = result + "<tor>"
    assert len(list(find_special_token(result, '<tor>'))) == tor_count
    return result

def add_bundle_tokens(input_string, special_token, num):

    # number of special tokens in input_string
    num_special_tokens = len(list(find_special_token(input_string, special_token)))

    # No special token -> return the raw
    if not num_special_tokens:
        return input_string
    
    result = ""
    index = 0
    while index < len(input_string):
        if input_string[index:index + len(special_token)] == special_token:
            result += special_token * num
            index += len(special_token)
        else:
            result += input_string[index]
            index += 1

    assert len(list(find_special_token(result, special_token))) == num_special_tokens * num
    return result

def make_instruction_for_mmamba(question, tor=None):
    
    if tor:
        qa_prompt = make_human_string(f"<s>[UNUSED_TOKEN_146]user\n{question}[UNUSED_TOKEN_145]",
                                    f"[UNUSED_TOKEN_146]rationale\n{tor}[UNUSED_TOKEN_145]\n</s>",
                                    split='\n')
    else:
        qa_prompt = make_human_string(f"<s>[UNUSED_TOKEN_146]user\n{question}[UNUSED_TOKEN_145]",
                                    f"[UNUSED_TOKEN_146]rationale\n"+"<tor>"*10+"[UNUSED_TOKEN_145]\n</s>",
                                    split='\n')
    return qa_prompt

def make_instruction_for_eval_meteor(question, dataset):
    system_prompt = "You should give helpful answer to user based on the rationale."
    
    if dataset != "mmmu" and dataset != "mathverse" and dataset != "hallusionbench" and dataset != "demo":
        question = "<image>" + question

    if dataset in ["sqa", "mmbench", "mmbench_cn", "mmbench_dev", "mmbench_cn_dev", "seed", "qbench", "ai2d", "mmstar"]:
        question = question + "\nAnswer with the option's letter from the given choices directly."

    elif dataset in ["vqav2", "gqa", "pope", "chartqa"]:
        question = question + "\nAnswer the question using a single word or phrase."

    elif dataset in ["vizwiz"]:
        question = question + "\nWhen the provided information is insufficient, respond with 'Unanswerable'. Answer the question using a single word or phrase."

    elif dataset in ["mmmu"]:
        if "A." in question:
            question = question + "\nAnswer with the option's letter from the given choices directly."
        else:
            question = question + "\nAnswer the question using a single word or phrase."
        
    elif dataset in ["hallusionbench"]:
        if "Please answer yes or no." not in question:
            question = question + "Please answer yes or no."

    qa_prompt = make_human_string("<s>"+"<tor>"*10+f"[UNUSED_TOKEN_146]system\n{system_prompt}[UNUSED_TOKEN_145]",
                                f"[UNUSED_TOKEN_146]user\n{question}[UNUSED_TOKEN_145]",
                                "[UNUSED_TOKEN_146]assistant\n",
                                split='\n')

    return qa_prompt


def make_human_string(*args, split):
    out = ''
    for i, arg in enumerate(args):
        out += arg
        if i != len(args)-1:
            out += split
    return out

def get_max_new_tokens(data_name):
    if data_name.lower() in ["mme", "pope", "sqa", "mmbench", "mmbench_cn", "mmbench_dev","mmbench_cn_dev", "seed", "qbench", "ai2d", "mmstar", "vqav2", "gqa", "chartqa", "hallusionbench", "textvqa", "mmmu"]:
        return 5
    if data_name.lower() in ["llava", "mm-vet"]:
        return 1024
    else:
        return 128

"""
Print Data Statistics
"""
def print_data_statistics(data):
    # name set
    name_set = {'caption',
                'instruction',
                'minigemini',
                'docdownstream',
                'docreason',
                'gllava',
                'mathvision',
                'mathinstruct',
                'mathplus'}
    caption = []
    instruction = []
    minigemini = []
    docdownstream = []
    docreason = []
    gllava = []
    mathvision = []
    mathinstruct = []
    mathplus = []
    for d in data:
        for name in name_set:
            if name in d['id']:
                eval(f'{name}.append(1)')
                break
    num_caption = sum(caption)
    num_instruction = sum(instruction)
    num_minigemini = sum(minigemini)
    num_docdownstream = sum(docdownstream)
    num_docreason = sum(docreason)
    num_gllava = sum(gllava)
    num_mathvision = sum(mathvision)
    num_mathinstruct = sum(mathinstruct)
    num_mathplus = sum(mathplus)

    total_len = num_caption + num_instruction + num_minigemini + \
    num_docdownstream + num_docreason + num_gllava + \
    num_mathvision +  num_mathinstruct + num_mathplus

    print('Meteor Dataset Structure Statistics')
    print(f'Total Length: {total_len}')
    print('--------------------------------------------')
    print(f'ShareGPT4V-Caption: {num_caption}')
    print(f'ShareGPT4V-Instruction: {num_instruction}')
    print(f'MiniGemini: {num_minigemini}')
    print(f'DocDownstream: {num_docdownstream}')
    print(f'DocReason: {num_docreason}')
    print(f'GLLaVA: {num_gllava}')
    print(f'MathVision: {num_mathvision}')
    print(f'MathInstruct: {num_mathinstruct}')
    print(f'MathPlus: {num_mathplus}')
    print('--------------------------------------------')
    print(f'Real-World Image: {num_caption + num_instruction}')
    print(f'Document & Chart & Diagram & Sign & Symbol: {num_minigemini + num_docdownstream + num_docreason}')
    print(f'Math: {num_gllava + num_mathvision + num_mathinstruct + num_mathplus}')
    print(f'     Math with Vision: {num_gllava + num_mathvision}')
    print(f'     Math with Text only: {num_mathinstruct + num_mathplus}')
    print('--------------------------------------------')
    print('')