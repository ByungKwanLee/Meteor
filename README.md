# <img src="figures/meteor_emoji.png" style="vertical-align: 0px;" :height="50px" width="30px">Meteor: Mamba-based traversal of rationale for Large Language and Vision Models

## 📰 News
<!-- - Meteor is now available in 🤗[Huggingface Models]().
- Curated 1.1M Question-Rationale-Answer Triples are now available in 🤗[Huggingface Datasets](). -->
- Meteor-7B is soon released in 🤗 Huggingface Models.
- Curated 1.1M Question-Rationale-Answer Triples are soon released in 🤗 Huggingface Datasets.
- Preprint of Meteor is soon uploaded in ArXiv.


Official PyTorch implementation code for realizing the technical part of *Mamba-based traversal of rationale (Meteor)* to improve numerous vision language performances with efficient model size. This code is developed from scratch. so I have been trying to improve the readibility and simplicity of the code, compared with LLaVA which has relatively complexly structured code.

The contributions of Meteor can be simply summarized as the following lists

- [x] Curated 1.1M Question-Rationale-Answer Triples.
- [x] Meteor is the efficient 7B model, compared with highly Larger LLVMs.
- [x] Meteor-7B acquires diverse capabilities, thereby showing surprising powerful vision language performances.


## 💡 Highlights
Open-source LLVMs with Standard Model Size

| LLVMs         | SQA-IMG | POPE |  MME |  MMB | MathVista | SEED-IMG | MM-Vet | LLaVA-W |
|---------------|:-------:|:----:|:----:|:----:|:---------:|:--------:|:------:|:-------:|
| Yi-VL-6B      |   71.7  | 82.5 | 1915 | 64.2 |    29.7   |  67.5  |  32.1  |   51.9  |
| LLaVA-NeXT-7B |   70.1  | 86.5 | 1851 | 69.6 |    34.6   |  70.2  |  43.9  |   72.3  |
| MM1-7B        |   72.6  | 86.6 | 1858 | 72.3 |    35.9   |  70.9  |  42.1  |    -    |
| Meteor-7B | **88.3**| **88.7** | **2229** | **82.9** |  **75.0** | **75.0** | **57.3** | **87.1** |

Open-source LLVMs with Large Model Sizes

| LLVMs             |   AI2D   |  ChartQA |    MME   |    MMB   | MathVista |  MM-Vet  |  LLaVA-W |
|-------------------|:--------:|:--------:|:--------:|:--------:|:---------:|:--------:|:--------:|
| InternVL1.5-40B   |   79.0   |   68.0   |   2175   |   82.2   |    47.7   |   48.9   |     -    |
| InternVL1.5-26B   | **80.7** | **83.8** |   2188   |   82.2   |  **53.5** | **62.8** |     -    |
| MM1-30B           |     -    |     -    |   2069   |   75.1   |    39.4   |   48.7   |     -    |
| MiniGemini-34B    |     -    |     -    |   2105   |   79.6   |    38.9   |   53.0   |     -    |
| MiniGemini-HD-34B |     -    |     -    |   2141   |   80.6   |    43.3   |   59.3   |     -    |
| LLaVA-NeXT-34B    |   74.9   |   68.7   |   2030   |   79.3   |    46.0   |   57.4   |   88.8   |
| LLaVA-NeXT-8B     |   71.6   |   69.5   |   1972   |   72.1   |    37.5   |     -    |   80.1   |
| LLaVA-NeXT-72B    |   77.4   |   77.0   |   2159   |   80.5   |    46.6   |     -    |   89.2   |
| LLaVA-NeXT-110B   |   80.4   |   80.4   |   2201   |   80.5   |    49.0   |     -    | **90.4** |
| Meteor-7B         |   77.9   |   74.9   | **2229** | **82.9** |    53.4   |   57.3   |   87.1   |

Closed-source LLVMs

| LLVMs        |  SQA-IMG |   AI2D   |  ChartQA |    MME   |    MMB   | MathVista | SEED-IMG |  MMStar  |
|--------------|:--------:|:--------:|:--------:|:--------:|:--------:|:---------:|:--------:|:--------:|
| Qwen-VL-Plus |   71.6   |   75.9   | **78.1** |   2183   |   67.0   |    71.6   |   72.7   |   39.7   |
| Gemini-Pro   |   80.1   |   73.9   |   74.1   |   1933   |   73.6   |    80.1   |   70.7   |   41.6   |
| GPT-4V       |   84.6   | **78.2** |   78.5   |   1927   |   77.0   |    84.6   |   69.1   |   46.1   |
| Meteor-7B    | **88.3** |   77.9   |   74.9   | **2229** | **82.9** |  **53.4** | **75.0** | **52.8** |


## 📋 Gathered & Curated Dataset Description
Gathered Total: 2130830, 2.1M
```shell
------------------------------
* Real-World Image: 755k
* Document & Chart & Diagram & Sign & Symbol: 627k
* Math: 747k
    - Math with Vision: 180k
    - Math with Text only: 566k
------------------------------

- ShareGPT4V-Caption [without SAM] (91021, 91k)
- ShareGPT4V-Instruction [Without few samples of OCR-VQA] (664703, 664k)
- MiniGemini-Instruction [DocVQA, ChartQA, DVQA, AI2D] (27670, 27k)
- DocDownstream (574268, 574k)
- DocReason (25877, 25k)
- GLLaVA-Align (60252, 60k)
- GLLaVA-QA (117205, 117k)
- MathVision (3040, 3k)
- MathInstruct [TextOnlyDataset] (262040, 262k)
- MathPlus [TextOnlyDataset] (304754, 304k)
```

Curated Total: 1059382, 1.1M
```shell
--------------------------------------------
Real-World Image: 338K
Document & Chart & Diagram & Sign & Symbol: 379K
Math: 342K
     Math with Vision: 165K
     Math with Text only: 177K
--------------------------------------------


- ShareGPT4V-Caption (72507, 73K)
- ShareGPT4V-Instruction (266072, 266K)
- MiniGemini-Instruction (26885, 27K)
- DocDownstream (298748, 299K)
- DocReason (53065, 53K)
- GLLaVA (162378, 162K)
- MathVision (2992, 3K)
- MathInstruct (81496, 81K)
- MathPlus (95239, 95K)

```


## 🚀 Download Training Datasets

We collect the following eight datasets. For MiniGemini, we selectively use data samples only for DocVQA, ChartQA, DVQA, and AI2D. Therefore, it is no need for you to download all data samples for MiniGemini.

* ShareGPT4V [[link](https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md)]
* MiniGemini [[link](https://github.com/dvlab-research/MiniGemini)]
* DocDownstream [[link](https://huggingface.co/datasets/mPLUG/DocDownstream-1.0)]
* DocReason [[link](https://huggingface.co/datasets/mPLUG/DocReason25K)]
* GLLaVA [[link](https://huggingface.co/datasets/Luckyjhg/Geo170K)]
* MathVision [[link](https://huggingface.co/datasets/mathvision/mathvision)]
* MathInstruct [[link](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)]
* MathPlus [[link](https://huggingface.co/datasets/TIGER-Lab/MATH-plus)]

Gathered Dataset Layout
```bash
Meteor_Dataset_Path
├── llava                                                       # ShareGPT4V
│   └── llava_pretrain                  
│       └── images                  
├── coco                                                        # ShareGPT4V
│   └── train2017                   
├── sam                                                         # ShareGPT4V
│   └── images                  
├── gqa                                                         # ShareGPT4V
│   └── images                  
├── ocr_vqa                                                     # ShareGPT4V
│   └── images                  
├── textvqa                                                     # ShareGPT4V
│   └── train_images                    
├── vg                                                          # ShareGPT4V
│   ├── VG_100K                 
│   └── VG_100K_2                   
├── share_textvqa                                               # ShareGPT4V
│   └── images                  
├── web-celebrity                                               # ShareGPT4V
│   └── images                  
├── web-landmark                                                # ShareGPT4V
│   └── images                  
├── wikiart                                                     # ShareGPT4V
│   └── images                  
├── share_textvqa                                               # ShareGPT4V
│   └── images                  
├── docvqa                                                      # MiniGemini
│   └── images                  
├── chartqa                                                     # MiniGemini
│   └── train                   
│       └── images                  
├── dvqa                                                        # MiniGemini
│   └── images                  
├── ai2d                                                        # MiniGemini
│   └── images                  
├── imgs                                                        # DocDownstream & DocReason
│   └── ChartQA
│   └── DUE_Benchmark
│       └── DeepForm
│       └── DocVQA
│       └── InfographicsVQA
│       └── KleisterCharity
│       └── TabFact
│       └── WikiTableQuestions
│   └── TextCaps
│   └── TextVQA
│   └── VisualMRC
├── geo3k                                                       # GLLaVA
|   └── train
├── geoqa_plus                                                  # GLLaVA
├── images                                                      # MathVision
|
├── sharegpt4v_instruct_gpt4-vision_cap100k.json                # ShareGPT4V-Caption
├── sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json  # ShareGPT4V-Instruction
├── train.jsonl                                                 # DocDownstream
├── detailed_explanation.jsonl                                  # DocReason
├── minigemini_instruction.json                                 # MiniGemini-Instruction
├── gllava_align.parquet                                        # GLLaVA-Align
├── gllava_qa.parquet                                           # GLLaVA-QA
├── mathvision.parquet                                          # MathVision
├── MathInstruct.json                                           # MathInstruct
└── mathplus.parquet                                            # MathPlus
```

## 📂 Evaluation Benchmarks

These are the list of evaluation datasets. If you completely download them, the dataset should be placed in the folder by the following below directory layout.

* Q-Bench [[link](https://github.com/Q-Future/Q-Bench)]
* SQA-IMG [[link](https://scienceqa.github.io/)]
* AI2D [[link](https://allenai.org/data/diagrams)]
* ChartQA [[link](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)]
* SEED [[link](https://github.com/AILab-CVC/SEED-Bench)]
* POPE [[link](https://github.com/RUCAIBox/POPE)]
* HallusionBench [[link](https://github.com/tianyi-lab/HallusionBench)]
* MME [[link](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)]
* MathVista [[link](https://github.com/lupantech/MathVista)]
* MMB [[link](https://github.com/open-compass/MMBench?tab=readme-ov-file)]
* MM-Vet [[link](https://github.com/yuweihao/MM-Vet)]
* LLaVA-W [[link](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild)]
* MMStar [[link](https://huggingface.co/datasets/Lin-Chen/MMStar)]
* MathVerse [[link](https://huggingface.co/datasets/AI4Math/MathVerse)]

Evaluation Dataset Directory Layout
```bash
Evaluation_Dataset_Path
├── LLVisionQA-QBench               # Q-Bench
├── ScienceQA                       # SQA-IMG
├── ai2d                            # AI2D
├── chartqa                         # ChartQA
├── SEED-Bench                      # SEED-IMG
├── POPE                            # POPE
├── HallusionBench                  # HallusionBench
├── MME_Benchmark_release_version   # MME
├── MathVista                       # MathVista
├── MMBench                         # MMB
├── mm-vet                          # MM-Vet
├── llava-bench-in-the-wild         # LLaVA Bench in the Wild
├── MMStar                          # MMStar
└── MathVerse                       # MathVerse
```
