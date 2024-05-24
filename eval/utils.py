import os
import re
import json
import openai
from typing import Dict
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from typing import Optional
from collections import defaultdict
from eval.mathvista.utilities import get_chat_response
from config import *
from copy import deepcopy

random.seed(42)

# SEED Question types
SEED_TYPES = {1: 'Scene Understanding', 2: 'Instance Identity', 3: 'Instance Location', 4: 'Instance Attributes', 5: 'Instances Counting', 6: 'Spatial Relation', 7: 'Instance Interaction', 8: 'Visual Reasoning', 9: 'Text Understanding'}

# Check for duplicated questions, items
def remove_duplicate(dataset, inputs, gen_answers):
    if dataset == "mme": 
        return inputs, gen_answers
    elif dataset == "pope": 
        questions = set()
        new_inputs, new_answers = [], []
        for i, a in zip(inputs, gen_answers):
            dup = i['id'], i['category']
            if dup in questions:
                continue
            questions.add(dup)
            new_inputs.append(i)
            new_answers.append(a)
    else:
        questions = set()
        new_inputs, new_answers = [], []
        for i, a in zip(inputs, gen_answers):
            if i['id'] in questions:
                continue
            questions.add(i['id'])
            new_inputs.append(i)
            new_answers.append(a)
    return new_inputs, new_answers

class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI
        copied from
        https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


class TextVQAAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()

    def _compute_answer_scores(self, raw_answers):
        """
        compute the accuracy (soft score) of human answers
        """
        answers = [self.answer_processor(a) for a in raw_answers]
        assert len(answers) == 10
        gt_answers = list(enumerate(answers))
        unique_answers = set(answers)
        unique_answer_scores = {}

        for unique_answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [
                    item for item in other_answers if item[1] == unique_answer
                ]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[unique_answer] = sum(accs) / len(accs)

        return unique_answer_scores

    def eval_pred_list(self, pred_list):
        pred_scores = []
        for entry in pred_list:
            pred_answer = self.answer_processor(entry["pred_answer"])
            unique_answer_scores = self._compute_answer_scores(entry["gt_answers"])
            score = unique_answer_scores.get(pred_answer, 0.0)
            pred_scores.append(score)

        accuracy = sum(pred_scores) / len(pred_scores)
        return accuracy

# MME
class MMEEvaluator:
    def divide_chunks(self, l, n=2):
        # looping till length l
        for i in range(0, len(l), n): 
            yield l[i:i + n]
        
        return 

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"

        return pred_label


    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict


    def process_result(self, results_dir):
        eval_type_dict = {
            "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
            "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
        }

        model_score_dict = dict()
        for eval_type, task_name_list in eval_type_dict.items():
           
            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                task_txt = os.path.join(results_dir, task_name + ".txt")
                lines = open(task_txt, 'r').readlines()
                chunk_lines = list(self.divide_chunks(lines)) # one image corresponds to two questions
                
                img_num = len(chunk_lines)
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []

                for img_items in chunk_lines:
                    assert len(img_items) == 2
                    img_correct_num = 0

                    for img_item in img_items:
                        img_name, question, gt_ans, pred_ans = img_item.split("\t")

                        gt_ans = gt_ans.lower()
                        pred_ans = pred_ans.lower()

                        assert gt_ans in ["yes", "no"] # gt can only be yes or no.

                        pred_ans = self.parse_pred_ans(pred_ans)
                        assert pred_ans in ["yes", "no", "other"]

                        gts.append(gt_ans)
                        preds.append(pred_ans)
                        
                        if gt_ans == pred_ans:
                            img_correct_num += 1
                        
                        if pred_ans not in ["yes", "no"]:
                            task_other_ans_num += 1

                    if img_correct_num == 2:
                        acc_plus_correct_num += 1

                # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                metric_dict["acc_plus"] = acc_plus
                
                
                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v*100
                
                task_score_dict[task_name] = task_score
                
                scores += task_score
            task_score_dict['total'] = scores
            model_score_dict[eval_type] = task_score_dict
        return model_score_dict

# For MMMU, convert all <image #> tokens to <image>
def replace_image_tokens(question):
    replaced = set()
    def replace_token(match):
        token = match.group(0)
        if token not in replaced:
            replaced.add(token)
            return '<image>'
        return token

    pattern = re.compile(r'<image\s\d+>')
    return pattern.sub(replace_token, question)

# For MMMU, count all <image #> tokens
def count_unique_image_tokens(string):
    pattern = r'<image\s\d+>'
    matches = re.findall(pattern, string)
    return len(set(matches))

# TextVQA    
def prompt_processor(self, prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()

# Convert answer to integer
def char_to_int(char):
    return ord(char.upper()) - ord('A')

# In case model does not output a single letter, find the choice in answer
def convert_to_choice(answer, candidates):
    options = ["A", "B", "C", "D", "E"]
    if answer in options:
        extracted_answer = answer
    elif len(answer) >= 2 and answer[0] in options and "." in answer:
        extracted_answer= answer[0]
    else:
        pattern = re.compile(r'The answer is ([A-Z]).')
        res = pattern.findall(answer)
        if len(res) == 1:
            extracted_answer = res[0]  # 'A', 'B', ...
        else:
            extracted_answer = "FAILED"

    if extracted_answer in options[:len(candidates)]:
        return options.index(extracted_answer)
    else:
        return -1
    
def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return -1

# Chart QA
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
    target: Target string.
    prediction: Predicted string.
    max_relative_change: Maximum relative change.

    Returns:
    Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                            target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()

# MME
def get_gt(data_path):
    ground_truth = {}
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if not os.path.isdir(category_dir):
            continue
        if os.path.exists(os.path.join(category_dir, 'images')):
            image_path = os.path.join(category_dir, 'images')
            qa_path = os.path.join(category_dir, 'questions_answers_YN')
        else:
            image_path = qa_path = category_dir
        assert os.path.isdir(image_path), image_path
        assert os.path.isdir(qa_path), qa_path
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            for line in open(os.path.join(qa_path, file)):
                question, answer = line.strip().split('\t')
                ground_truth[(category, file, question)] = answer
    return ground_truth

# Chart QA
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
    target: Target string.
    prediction: Predicted string.
    max_relative_change: Maximum relative change.

    Returns:
    Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%'))
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                            target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()
        
def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)

# POPE
def eval_pope(answers, label_file):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['answer'].lower()

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['answer'] = 'no'
        else:
            answer['answer'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['answer'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc

# Eval GQA
# book to float
def toScore(b):
    return float(1 if b else 0)

# Compute average of a list
def avg(l):
    if len(l) == 0:
        return 0
    return float(sum(l)) / len(l)

def eval_gqa(predictions, questions):
    # Initialize data structure to track all metrics: e.g. accuracy, validity and plausibility, as well as 
    # accuracy per question type, length and number of reasoning steps.
    scores = {
        "accuracy": [], # list of accuracies per question (1 if correct else 0). Will be averaged ultimately.
        "binary": [], # list of accuracies per a binary question (1 if correct else 0). Will be averaged ultimately.
        "open": [], # list of accuracies per an open question (1 if correct else 0). Will be averaged ultimately.
        "validity": [], # list of validity per question (1 if valid else 0).
        "plausibility": [], # list of plausibility per question (1 if plausible else 0). 
        "consistency": [], # list of consistency scores for entailed questions.
        "accuracyPerStructuralType": defaultdict(list), # list of question accuracies for each structural type (e.g. compare, logic questions).
        "accuracyPerSemanticType": defaultdict(list), # list of question accuracies for each semantic type (e.g. questions about an object, an attribute, a relation).
        "accuracyPerLength": defaultdict(list), # list of question accuracies per question's word number.
        "accuracyPerSteps": defaultdict(list), # list of question accuracies per question's reasoning length (steps number).
        "grounding": [] # list of grounding scores for each question.
    }

    # Initialize golden and predicted histograms per each question group. Used to compute the distribution metric.
    dist = {
        "gold": defaultdict(lambda: defaultdict(int)),
        "predicted": defaultdict(lambda: defaultdict(int))
    }

    ##### Question lengths - words numbers and reasoning steps number
    ##########################################################################################

    # Compute question length (words number)
    def getWordsNum(question):
        return len(question["question"].split())

    # Compute number of reasoning steps (excluding the final "querying" step which doesn't increase effective reasoning length)
    def getStepsNum(question):
        return len([c for c in question["semantic"] if not (any([o in "{}: {}".format(c["operation"], c["argument"]) 
            for o in ["exist", "query: name", "choose name"]]))])

    ##### Main score computation 
    ##########################################################################################

    # Loop over the questions and compute mterics
    for qid, question in questions.items():
        gold = question["answer"]
        if qid not in predictions:
            continue
        predicted = predictions[qid].lower()

        correct = (predicted == gold)
        score = toScore(correct)

        wordsNum = getWordsNum(question)
        stepsNum = getStepsNum(question)
        
        # Compute scores over the balanced dataset (more robust against cheating by making educated guesses)
        if question["isBalanced"]:
            # Update accuracy
            scores["accuracy"].append(score)
            scores["accuracyPerLength"][wordsNum].append(score)
            scores["accuracyPerSteps"][stepsNum].append(score)
            scores["accuracyPerStructuralType"][question["types"]["structural"]].append(score)
            scores["accuracyPerSemanticType"][question["types"]["semantic"]].append(score)
            answerType = "open" if question["types"]["structural"] == "query" else "binary"
            scores[answerType].append(score)
    
            # Update histograms for gold and predicted answers
            globalGroup = question["groups"]["global"]
            if globalGroup is not None:
                dist["gold"][globalGroup][gold] += 1
                dist["predicted"][globalGroup][predicted] += 1

    # Average scores over all questions (in the balanced dataset) and print scores
    metrics = [
        "binary",
        "open",
        "accuracy",
        "consistency",
        "validity",
        "plausibility",
        "grounding",
    ]

    detailedMetrics = [
        ("accuracyPerStructuralType", "Accuracy / structural type"), 
        ("accuracyPerSemanticType", "Accuracy / semantic type"), 
        ("accuracyPerSteps", "Accuracy / steps number"),
        ("accuracyPerLength", "Accuracy / words number") 
    ]

    subMetrics = {
        "attr": "attribute",
        "cat": "category",
        "global": "scene",
        "obj": "object",
        "rel": "relation" 
    }
    # average
    for k in metrics:
        if isinstance(scores[k], list):
            scores[k] = avg(scores[k]) * 100

    for k, _ in detailedMetrics:
        for t in scores[k]:
            scores[k][t] = avg(scores[k][t]) * 100, len(scores[k][t])

    # print
    print("")
    for m in metrics:
        # skip grounding and consistency scores if not requested
        if m == "grounding":
            continue
        if m == "consistency":
            continue

        # print score
        print("{title}: {score:.2f}{suffix}".format(title = m.capitalize(), score = scores[m], 
            suffix = " (lower is better)" if m == "distribution" else "%"))

    for m, mPrintName in detailedMetrics:
        print("")
        # print metric title
        print("{}:".format(mPrintName))
        
        for t in sorted(list(scores[m].keys())):
            # set sub-metric title
            tName = t
            if isinstance(scores[k], list):
                tName = subMetrics.get(t, t).capitalize()

            # print score
            print("  {title}: {score:.2f}{suffix} ({amount} questions)".format(title = tName, 
                score = scores[m][t][0], suffix = "%", amount = scores[m][t][1]))    
            
    return scores

# MMMU
DOMAIN_CAT2SUB_CAT = {
  'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'],
  'Business': ['Accounting', 'Economics', 'Finance', 'Manage','Marketing'],
  'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics',],
  'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
  'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'],
  'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
}


CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

"""Response Parsing and Evaluation for various models"""

# ----------- Process Multi-choice -------------
def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack: 
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

# ----------- Process Open -------------
def check_is_number(string):
    """
    Check if the given string a number.
    """
    try:
        float(string.replace(',', ''))
        return True
    except ValueError:
        # check if there's comma inside
        return False

def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    """
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(',', '')
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else: # it's likely to be a string
        # lower it 
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "] # avoid trivial matches
        return [string]

def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    """
    # Pattern for numbers with commas
    pattern_commas = r'-?\b\d{1,3}(?:,\d{3})+\b'
    # Pattern for scientific notation
    pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
    # Pattern for simple numbers without commas
    pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])'

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbers
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers

def parse_open_response(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """
    # content = content.strip("\n").strip(".").strip(" ")
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r'\.\s(?=[A-Z])|\n', response)
        indicators_of_keys = ['could be ', 'so ', 'is ',
                            'thus ', 'therefore ', 'final ', 'answer ', 'result ']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(['='])
            shortest_key_response = None # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()
                    # key_responses.append(resp.split(indicator)[1].strip())
            
            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0: # did not found any
            return [response]
        return key_responses
    # pdb.set_trace()
    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy() # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list

# ----------- Evaluation -------------

def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    """
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else: # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct

def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    """
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i: # pred is already normalized in parse response phase
        if isinstance(pred, str): # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else: # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct

# ----------- Batch Evaluation -------------
def evaluate(samples):
    """
    Batch evaluation for multiple choice and open questions.
    """
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample['answer']
        pred_i = sample['parsed_pred']
        if sample['question_type'] == 'multiple-choice':
            correct = eval_multi_choice(gold_i, pred_i)
        else: # open question
            correct = eval_open(gold_i, pred_i)

        if correct:
            judge_dict[sample['id']] = 'Correct'
            pred_correct += 1
        else:
            judge_dict[sample['id']] = 'Wrong'

    if len(samples) == 0:
        return {'acc': 0}
    return judge_dict, {'acc': pred_correct / len(samples)}



# ----------- Calculate Accuracy -------------
def calculate_ins_level_acc(results: Dict):
    """Calculate the instruction level accuracy for given Subject results"""
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results['acc'] * cat_results['num_example']
        ins_num += cat_results['num_example']
    if ins_num == 0:
        return 0
    return acc / ins_num
    
def eval_mathverse(output_dir, results, extract_file, score_file):
    openai.api_key = OPENAI_KEY
    
    demo_prompt_extract = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.

1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)

2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D

3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)

4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null

5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3

6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1

7.
"""
    demo_prompt_score = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.

[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0

[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0

[Question]: Given the graph of the ellipse that intersects with x-axis at 9 and -9 and with y-axis at 3 and -3, determine its equation.A. \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1 B. Can not determine.\n
[Standard Answer]: A
[Model_answer] : \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1
Judgement: 1

[Question]: {question}
[Standard Answer]: {gt}
[Model_answer] : {extraction}
Judgement: """
    
    def create_extract_prompt(demo_prompt, response, inst):
        demo_prompt = demo_prompt.strip()
        test_prompt = f"Model response: '{response}'\nExtracted Answer: "
        full_prompt = f"{demo_prompt}\n\n{test_prompt}"
        return full_prompt
    
    def create_scoring_prompt(demo_prompt, inst):
        demo_prompt = demo_prompt.strip()
        full_prompt = demo_prompt.format(question = inst['question'], gt=inst['answer'], extraction=inst['extraction'])
        return full_prompt


    def extract_answer(response, inst, api_key):
        # general extraction
        try:
            full_prompt = create_extract_prompt(demo_prompt_extract, response, inst)
            extraction = get_chat_response(full_prompt, api_key)
            return extraction
        except Exception as e:
            print(e)
            print(f"Error in extracting answer for {response}")
        return ""
    
    def match_answer(inst, api_key):
        try:
            full_prompt = create_scoring_prompt(demo_prompt_score, inst)
            extraction = get_chat_response(full_prompt, api_key)
            return extraction.replace("Judgement:", "").strip()
        except Exception as e:
            print(e)
            print(f"Error in matching answer")

        return ""
    
    save_results = []
    score_dict = defaultdict(lambda: defaultdict(list))
    score_version_dict = defaultdict(list)
    
    for i, inst in enumerate(tqdm(results)):
        response = inst['model_answer']
        extraction = extract_answer(response, inst, OPENAI_KEY)
        inst['extraction'] = extraction.replace('Extracted Answer: ', '').strip()
        
        judgement = match_answer(inst, OPENAI_KEY)
        while True:
            if judgement.strip() not in ['0', '1']:
                print('Wrong return format: ', judgement)
                judgement = match_answer(inst, OPENAI_KEY)
            else:
                inst['judgement'] = int(judgement)
                break

        save_results.append(inst)

        score_dict[inst['metadata']['subject']][inst['metadata']['subfield']].append(inst['judgement'])
        score_version_dict[inst['problem_version']].append(inst['judgement'])
    
    results_file = os.path.join(output_dir, extract_file)
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=4)
        
    print(f"Save MathVerse Results at {results_file}")
    
    save_json = {}
    # version level acc
    total_cnt, right_cnt = 0, 0
    for version in score_version_dict:
        version_total_cnt = len(score_version_dict[version])
        version_right_cnt = len([inst for inst in score_version_dict[version] if inst == 1])
        total_cnt += version_total_cnt
        right_cnt += version_right_cnt
        print(f"{version} Acc: {(version_right_cnt/version_total_cnt):.3f}")
        save_json[version] = f"{(version_right_cnt/version_total_cnt):.3f}"

    print(f"Acc: {(right_cnt/total_cnt):.3f}")
    
    save_json["Total Acc"] = f"{(right_cnt/total_cnt):.3f}"
    
    scores_file = os.path.join(output_dir, score_file)
    with open(scores_file, 'w') as f:
        json.dump(save_json, f, indent=4)
    
    
def eval_mmstar(eval_file, output_dir, score_file):
    MMStar_score_l2 = {
        'coarse perception': {
            'image scene and topic': 0,
            'image style & quality': 0,
            'image emotion': 0
        },
        'fine-grained perception': {
            'object counting': 0,
            'recognition': 0,
            'localization': 0
        },
        'instance reasoning': {
            'single-instance reasoning': 0,
            'cross-instance attribute reasoning': 0,
            'cross-instance relation reasoning': 0
        },
        'logical reasoning': {
            'code & sequence reasoning': 0,
            'diagram reasoning': 0,
            'common reasoning': 0
        },
        'science & technology': {
            'biology & chemistry & physics': 0,
            'electronics & energy & mechanical eng.': 0,
            'geography & earth science & agriculture': 0
        },
        'math': {
            'geometry': 0,
            'numeric commonsense and calculation': 0,
            'statistical reasoning': 0
        },
    }
    MMStar_counter = deepcopy(MMStar_score_l2)

    data = eval_file
    lt = len(data)
    lines = [data.iloc[i] for i in range(lt)]
    for i in tqdm(range(len(lines))):
        line = lines[i]
        predict = str(line['prediction'])
        answers = str(line['answer'])
        # ori_bench = str(line['bench'])
        category = str(line['category'])
        l2_category = str(line['l2_category'])
        MMStar_counter[category][l2_category] += 1

        answer = answers.lower().strip().replace('\n', ' ')
        predict = predict.lower().strip().replace('\n', ' ')
        # if ori_bench == 'MathVista' and answer not in ['a', 'b', 'c', 'd']:
        #     if answer in predict:
        #         MMStar_score_l2[category][l2_category] += 1
        # else:
        try:
            if answer == predict[0]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0] == '(' and answer == predict[1]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0:7] == 'option ' and answer == predict[7]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0:14] == 'the answer is ' and answer == predict[14]:
                MMStar_score_l2[category][l2_category] += 1
        except Exception as e:
            pass

    MMStar_score = {}
    MMStar_score['final score'] = 0
    for k, v in MMStar_score_l2.items():
        MMStar_score[k] = 0
        for l2_k, l2_v in v.items():
            if float(MMStar_counter[k][l2_k]) == 0:
                MMStar_score[f'{k}({l2_k})'] = 0
            else:
                MMStar_score[f'{k}({l2_k})'] = float(l2_v) / \
                    float(MMStar_counter[k][l2_k])
            MMStar_score[k] += l2_v
        MMStar_score['final score'] += MMStar_score[k]
        MMStar_score[k] = float(MMStar_score[k]) / 250.0
    MMStar_score['final score'] = float(MMStar_score['final score']) / 1500.0

    score_pth = os.path.join(output_dir, score_file)
    with open(score_pth, 'w') as f:
        json.dump(MMStar_score, f, indent=4)
        
    print(
        f'MMStar_eval successfully finished evaluating {eval_file}, results saved in {score_pth}')
    print('Score: ')
    for key, value in MMStar_score.items():
        print('{}:{}'.format(key, value))
