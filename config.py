# OpenAI Key
OPENAI_KEY = ""

# Dataset root
DATASET_ROOT=""

# Pre Meteor Dataset
METEOR_DATASET= "Meteor.json" 

# Various json and parquet files
SHAREGPT4V_CAPTION = "sharegpt4v_instruct_gpt4-vision_cap100k.json"
SHAREGPT4V_INSTRUCTION = "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"
MINIGEMINI_INSTRUCTION = "minigemini_instruction.json"
DOCDOWNSTREAM = 'train.jsonl'
DOCREASON = 'detailed_explanation.jsonl'
GLLAVA_ALIGN = "gllava_align.parquet"
GLLAVA_QA = "gllava_qa.parquet"
MATHVISION = "mathvision.parquet"
MATHINSTRUCT = "MathInstruct.json"
MATHPLUS = "mathplus.parquet"

# Json files for Evaluation
VQAV2 = "VQAv2/v2_OpenEnded_mscoco_test2015_questions.json"
GQA = "gqa/testdev_balanced_questions.json"
SQA = "ScienceQA/problems.json"
SQA_SPLIT = "ScienceQA/pid_splits.json"
VIZWIZ = "VizWiz/test.json"
TEXTVQA = "TextVQA/llava_textvqa_val_v051_ocr.json"
TEXTVQA_ANNOTATIONS = "TextVQA/TextVQA_0.5.1_val.json"
POPE_POPULAR = "POPE/coco_pope_popular.json"
POPE_ADVERSARIAL = "POPE/coco_pope_adversarial.json"
POPE_RANDOM = "POPE/coco_pope_random.json"
MME = "MME_Benchmark_release_version/llava_mme.json"
MME_DIR = "MME_Benchmark_release_version"
MMBENCH = "MMBench/MMBench_TEST_EN_legacy.tsv"
MMBENCH_CN = "MMBench/MMBench_TEST_CN_legacy.tsv"
MMBENCH_DEV = "MMBench/mmbench_dev_20230712.tsv"
MMBENCH_CN_DEV = "MMBench/mmbench_dev_cn_20231003.tsv"
QBENCH = "LLVisionQA-QBench/llvisionqa_dev.json"
QBENCH_CN = "LLVisionQA-QBench/质衡-问答-验证集.json"
MMVET = "mm-vet/mm-vet.json"
MMMU = "MMMU/*/validation*"
MATHVISTA = "MathVista/testmini-00000-of-00001-725687bf7a18d64b.parquet"
AI2D = "ai2d/ai2d_test.json"
HALLUSIONBENCH = "HallusionBench/HallusionBench.json"
CHARTQA = "chartqa/test/test_augmented.json"
SEED = "SEED-Bench/SEED-Bench.json"
LLAVA = "llava-bench-in-the-wild/questions.jsonl"
# BLINK =
MATHVERSE = "MathVerse/testmini.json"
MATHVERSE_TEXT_ONLY = "MathVerse/testmini_text_only.json"
MMSTAR = "MMStar/mmstar.parquet"

# Available evaluation datasets
EVAL_DATASETS = ["qbench", "sqa", "ai2d", "chartqa", "seed", "pope", "hallusionbench", "mme", \
                 "mathvista", "mmbench", "mmbench_cn", "mmvet", "llava", "mmstar", "mathverse"]

