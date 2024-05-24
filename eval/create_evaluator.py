import os
import re
import json
import shortuuid
import numpy as np
import pandas as pd
from config import *
from collections import defaultdict
from eval.utils import *

class BaseEvaluator:
    def __init__(self):
        super(BaseEvaluator, self).__init__()

        # Create evaluation results folder
        self.save_dir = os.path.join(DATASET_ROOT, "eval_results")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def reset(self):
        # Reset results for new dataset evaluation
        self.gen_answers = []
        self.inputs = []
    
    def process(self, inputs, outputs):
        # Merge results
        self.inputs.extend(inputs)
        self.gen_answers.extend(outputs)

class Evaluator(BaseEvaluator):
    def __init__(self):
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
              
        super().__init__()
    
    def evaluate(self, model, dataset, accel):

        # gathering all gpu to one device
        self.inputs = accel.gather_for_metrics(self.inputs)
        self.gen_answers = accel.gather_for_metrics(self.gen_answers)
        
        if accel.is_main_process:
            # check for duplicates
            self.inputs, self.gen_answers = remove_duplicate(dataset, self.inputs, self.gen_answers)

            # Select evaluation for dataset
            if dataset == "vqav2":
                return self.evaluate_vqa(model, accel)
            elif dataset == "gqa":
                return self.evaluate_gqa(model, accel)
            elif dataset == "sqa":
                return self.evaluate_sqa(model, accel)
            elif dataset == "vizwiz":
                return self.evaluate_vizwiz(model, accel)
            elif dataset == "textvqa":
                return self.evaluate_textvqa(model, accel)
            elif dataset == "pope":
                return self.evaluate_pope(model, accel)
            elif dataset == "mme":
                return self.evaluate_mme(model, accel)
            elif dataset == "mmbench":
                return self.evaluate_mmbench(model, accel)
            elif dataset == "mmbench_dev":
                return self.evaluate_mmbench_dev(model, accel)
            elif dataset == "mmbench_cn":
                return self.evaluate_mmbench_cn(model, accel)
            elif dataset == "mmbench_cn_dev":
                return self.evaluate_mmbench_cn_dev(model, accel)
            elif dataset == "qbench":
                return self.evaluate_qbench(model, accel)
            elif dataset == "mm-vet":
                return self.evaluate_mmvet(model, accel)
            elif dataset == "mmmu":
                return self.evaluate_mmmu(model, accel)
            elif dataset == "mathvista":
                return self.evaluate_mathvista(model, accel)
            elif dataset == "ai2d":
                return self.evaluate_ai2d(model, accel)
            elif dataset == "hallusionbench":
                return self.evaluate_hallusionbench(model, accel)
            elif dataset == "chartqa":
                return self.evaluate_chartqa(model, accel)
            elif dataset == "seed":
                return self.evaluate_seed(model, accel)
            elif dataset == "llava":
                return self.evaluate_llava(model, accel)
            elif dataset == "blink":
                return self.evaluate_blink(model, accel)
            elif dataset == "mathverse":
                return self.evaluate_mathverse(model, accel)
            elif dataset == "mmstar":
                return self.evaluate_mmstar(model, accel)
            else:
                raise ValueError(
                    f'{dataset} is not an available dataset.')
        else:
            return None
        
    def evaluate_vqa(self, model, accel):
        # VQAv2 Evaluation for EvalAI server
        pred_answers = [{'question_id': inputs['id'], 'answer': answer} for inputs, answer in zip(self.inputs, self.gen_answers)]
        pred_pth = os.path.join(self.save_dir, f"{model}_vqav2_results.json")
        json.dump(pred_answers, open(pred_pth, "w"))
        accel.print(f"Finished evaluating VQAv2. Evaluate the result file saved to {pred_pth} on EvalAI server.")
        return 
    
    def evaluate_gqa(self, model, accel):
        # GQA Evaluation
        pred_answers = {inputs['id']: answer for inputs, answer in zip(self.inputs, self.gen_answers)}
        # pred_answers = [{'question_id': inputs['id'], 'answer': answer} for inputs, answer in zip(self.inputs, self.gen_answers)]
        pred_pth = os.path.join(self.save_dir, f"{model}_gqa_results.json")
        json.dump(pred_answers, open(pred_pth, "w"))
        accel.print("GQA Results:")
        results = eval_gqa(pred_answers, json.load(open(os.path.join(DATASET_ROOT, GQA))))
        return results['accuracy']
    
    def evaluate_sqa(self, model, accel):
        # SQA Evaluation
        pred_answers = [{'question_id': inputs['id'], 'answer': convert_to_choice(answer, inputs['candidates']), 'gt': inputs['gt']} for inputs, answer in zip(self.inputs, self.gen_answers)]
        pred_pth = os.path.join(self.save_dir, f"{model}_sqa_results.json")
        json.dump(pred_answers, open(pred_pth, "w"))

        # Compute accuracy
        results = [(answer['answer'] == answer['gt']) for answer in pred_answers]
        accel.print (f"SQA Accuracy: {np.mean(results)*100} %")
        return np.mean(results)*100
    
    def evaluate_vizwiz(self, model, accel):
        # VizWiz Evaluation
        evaluator = EvalAIAnswerProcessor()
        pred_answers = [{'image': inputs['id'], 'answer': evaluator(answer)} for inputs, answer in zip(self.inputs, self.gen_answers)]
        pred_pth = os.path.join(self.save_dir, f"{model}_vizwiz_results.json")
        json.dump(pred_answers, open(pred_pth, "w"))
        accel.print(f"Finished evaluating VizWiz. Evaluate the result file saved to {pred_pth} on EvalAI server.")
        return 
    
    def evaluate_textvqa(self, model, accel):
        # TextVQA Evaluation
        pred_answers = [{'question_id': inputs['id'], 'pred_answer': answer, 'question': inputs['question'], 'gt_answers': inputs['gt']} for inputs, answer in zip(self.inputs, self.gen_answers)]
        pred_pth = os.path.join(self.save_dir, f"{model}_textvqa_results.json")
        json.dump(pred_answers, open(pred_pth, "w"))
            
        evaluator = TextVQAAccuracyEvaluator()
        results = evaluator.eval_pred_list(pred_answers)*100
        accel.print (f"TextVQA Accuracy: {results} %")
        return results
    
    def evaluate_pope(self, model, accel):
        # POPE Evaluation
        pred_answers = [{'question_id': inputs['id'], 'answer': answer, 'question': inputs['question'], 'category': inputs['category']} for inputs, answer in zip(self.inputs, self.gen_answers)]
        pred_pth = os.path.join(self.save_dir, f"{model}_pope_results.json")
        json.dump(pred_answers, open(pred_pth, "w"))

        pope_results = {}
        pope_results['adversarial'] = None
        pope_results['popular'] = None
        pope_results['random'] = None

        categories = ['adversarial', 'popular', 'random']
        files = [POPE_ADVERSARIAL, POPE_POPULAR, POPE_RANDOM]

        for category, file in zip(categories, files):
            cur_answers = [x for x in pred_answers if x['category'] == category]
            cur_answers = sorted(cur_answers, key=lambda x:x["question_id"])
            pope_results[category] = eval_pope(cur_answers, os.path.join(DATASET_ROOT, file))
        accel.print (f"POPE Adversarial Accuracy: {pope_results['adversarial']} %")
        accel.print (f"POPE Popular Accuracy: {pope_results['popular']} %")
        accel.print (f"POPE Random Accuracy: {pope_results['random']} %")
        return pope_results
    
    def evaluate_mme(self, model, accel):
        # MME Evaluation
        pred_answers = [{'question_id': inputs['id'], 'answer': answer,  "question": inputs['question'], 'category': inputs['category']} for inputs, answer in zip(self.inputs, self.gen_answers)]
        pred_pth = os.path.join(self.save_dir, f"{model}_mme_results.json")
        json.dump(pred_answers, open(pred_pth, "w"))

        ground_truth = get_gt(data_path=os.path.join(DATASET_ROOT, MME_DIR))
        result_dir = os.path.join(self.save_dir, 'mme')
        os.makedirs(result_dir, exist_ok=True)
        results = defaultdict(list)

        for answer in pred_answers:
            file = answer['question_id'].split('/')[-1].split('.')[0] + '.txt'
            results[answer['category']].append((file, answer['question'], answer['answer']))

        for category, cate_tups in results.items():
            with open(os.path.join(result_dir, f'{category}.txt'), 'w') as fp:
                questions = set() # check for duplicates
                for file, prompt, answer in cate_tups:
                    if 'Answer the question using a single word or phrase.' in prompt:
                        prompt = prompt.replace('Answer the question using a single word or phrase.', '').strip()
                    if 'Please answer yes or no.' not in prompt:
                        prompt = prompt + ' Please answer yes or no.'
                        if (category, file, prompt) not in ground_truth:
                            prompt = prompt.replace(' Please answer yes or no.', '  Please answer yes or no.')
                    gt_ans = ground_truth[category, file, prompt]
                    dup = file, prompt, gt_ans
                    tup = file, prompt, gt_ans, answer
                    if dup in questions:
                        continue
                    questions.add(dup)
                    fp.write('\t'.join(tup) + '\n')

        evaluator = MMEEvaluator()
        scores = evaluator.process_result(result_dir)
        accel.print("MME Scores:")
        accel.print(scores)
        for eval_type, eval_scores in scores.items():
            accel.print("===========", eval_type, "===========")
            accel.print("total score:", eval_scores['total'], "\n")
            for task_name, score in eval_scores.items():
                accel.print("\t", task_name, " score:", score)
            accel.print("\n")
        return scores
    
    def evaluate_mmbench(self, model, accel):
        # MMBench Evaluation
        df = pd.read_table(os.path.join(DATASET_ROOT, MMBENCH))
        cur_df = df.copy()
        cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
        cur_df.insert(6, 'prediction', None)
        for inputs, answer in zip(self.inputs, self.gen_answers):
            cur_df.loc[df['index'] == inputs['id'], 'prediction'] = answer
        pred_pth = os.path.join(self.save_dir, f"{model}_mmbench_results.xlsx")
        cur_df.to_excel(pred_pth, index=False, engine='openpyxl')
        accel.print(f"Finished evaluating MMBench. Change {pred_pth} name to submission.xlsx and evaluate the result file saved to {pred_pth} on OpenCompass server.")
        return 
    
    def evaluate_mmbench_dev(self, model, accel):
        # MMBench Dev Evaluation
        df = pd.read_table(os.path.join(DATASET_ROOT, MMBENCH_DEV))
        cur_df = df.copy()
        cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
        cur_df.insert(6, 'prediction', None)
        for inputs, answer in zip(self.inputs, self.gen_answers):
            cur_df.loc[df['index'] == inputs['id'], 'prediction'] = answer[0]
        pred_pth = os.path.join(self.save_dir, f"{model}_mmbench_dev_results.xlsx")
        cur_df.to_excel(pred_pth, index=False, engine='openpyxl')
        accuracy = (cur_df['prediction'] == cur_df['answer']).mean()
        accel.print(f'MMBench_dev Accuracy: {accuracy:.2%}')
        return 
    
    def evaluate_mmbench_cn(self, model, accel):
        # MMBench_CN Evaluation
        df = pd.read_table(os.path.join(DATASET_ROOT, MMBENCH_CN))
        cur_df = df.copy()
        cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
        cur_df.insert(6, 'prediction', None)
        for inputs, answer in zip(self.inputs, self.gen_answers):
            cur_df.loc[df['index'] == inputs['id'], 'prediction'] = answer
        pred_pth = os.path.join(self.save_dir, f"{model}_mmbench_cn_results.xlsx")
        cur_df.to_excel(pred_pth, index=False, engine='openpyxl')
        accel.print(f"Finished evaluating MMBench_CN. Change {pred_pth} name to submission.xlsx and evaluate the result file saved to {pred_pth} on OpenCompass server.")
        return 
    
    def evaluate_mmbench_cn_dev(self, model, accel):
        # MMBench_CN Dev Evaluation
        df = pd.read_table(os.path.join(DATASET_ROOT, MMBENCH_CN_DEV))
        cur_df = df.copy()
        cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
        cur_df.insert(6, 'prediction', None)
        for inputs, answer in zip(self.inputs, self.gen_answers):
            cur_df.loc[df['index'] == inputs['id'], 'prediction'] = answer[0]
        pred_pth = os.path.join(self.save_dir, f"{model}_mmbench_cn_dev_results.xlsx")
        cur_df.to_excel(pred_pth, index=False, engine='openpyxl')
        accuracy = (cur_df['prediction'] == cur_df['answer']).mean()
        accel.print(f'MMBench_CN_dev Accuracy: {accuracy:.2%}')
        return 
    
    def evaluate_qbench(self, model, accel):
        # QBench Evaluation
        pred_answers = [{'id': inputs['id'], 'answer': convert_to_choice(answer, inputs['candidates']), 'gt': inputs['gt'], 'candidates': inputs['candidates']} for inputs, answer in zip(self.inputs, self.gen_answers)]
        pred_pth = os.path.join(self.save_dir, f'{model}_qbench_results.jsonl')
        with open(pred_pth, "w") as pf:
            pf.write(json.dumps(pred_answers) + "\n")

        results = [(pred['candidates'][pred['answer']] == pred['gt']) for pred in pred_answers]
        accel.print (f"QBench Accuracy: {np.mean(results)*100} %")
        return np.mean(results)*100
    
    def evaluate_mmvet(self, model, accel):
        # MM-Vet Evaluation
        cur_result = {f"{inputs['id']}": answer for inputs, answer in zip(self.inputs, self.gen_answers)}
        pred_pth = os.path.join(self.save_dir, f'{model}_mmvet_results.json')
        with open(pred_pth, 'w') as f:
            json.dump(cur_result, f, indent=2)
        
        accel.print(f"Finished evaluating MM-Vet. Evaluate the result file saved to {pred_pth}.")
        return 
    
    def evaluate_mmmu(self, model, accel):
        # MMMU Evaluation
        predictions = {inputs['id']: answer for inputs, answer in zip(self.inputs, self.gen_answers)}
        answers = {inputs['id']: {'ground_truth': inputs['gt'], 'question_type': inputs['question_type']} for inputs, answer in zip(self.inputs, self.gen_answers)}
        pred_pth = os.path.join(self.save_dir, f'{model}_mmmu_results.json')
        with open(pred_pth, "w") as f:
            json.dump(predictions, f, indent=2)
        ans_pth = os.path.join(self.save_dir, 'mmmu_answers.json')
        with open(ans_pth, "w") as pf:
            json.dump(answers, pf, indent=2)

        # group by category
        output_dict_w_cat = {}
        for data_id, parsed_pred in predictions.items():
            category = "_".join(data_id.split("_")[1:-1])
            if category not in output_dict_w_cat:
                output_dict_w_cat.update({category: {}})
            output_dict_w_cat[category].update({data_id: parsed_pred})

        # group by category
        answer_dict_w_cat = {}
        for data_id, parsed_pred in answers.items():
            category = "_".join(data_id.split("_")[1:-1])
            if category not in answer_dict_w_cat:
                answer_dict_w_cat.update({category: {}})
            answer_dict_w_cat[category].update({data_id: parsed_pred})

        evaluation_result = {}

        for category in CAT_SHORT2LONG.values():
            accel.print("Evaluating: {}".format(category))
            # get cat_outputs and cat_answers
            try:
                cat_outputs = output_dict_w_cat[category]
                cat_answers = answer_dict_w_cat[category]
            except KeyError:
                accel.print("Skipping {} for not found".format(category))
                continue

            exampels_to_eval = []
            for data_id, parsed_pred in cat_outputs.items():
                question_type = cat_answers[data_id]['question_type']
                if question_type != 'multiple-choice':
                    parsed_pred = parse_open_response(parsed_pred) # mainly for type consistency (make it number, etc.)
                else:
                    parsed_pred = parsed_pred

                exampels_to_eval.append({
                    "id": data_id,
                    "question_type": question_type,
                    "answer": cat_answers[data_id]['ground_truth'],
                    "parsed_pred": parsed_pred
                })

            judge_dict, metric_dict = evaluate(exampels_to_eval)
            metric_dict.update({"num_example": len(exampels_to_eval)})

            evaluation_result[category] = metric_dict

        printable_results = {}
        # add domain Subject
        for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
            in_domain_cat_results = {}
            for cat_name in in_domain_cats: # use the order in DOMAIN_CAT2SUB_CAT
                if cat_name in evaluation_result.keys():
                    in_domain_cat_results[cat_name] = evaluation_result[cat_name]
                else:
                    pass
            in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
            in_domain_data_num = sum([cat_results['num_example'] for cat_results in in_domain_cat_results.values()])
            printable_results['Overall-' + domain] = {"num": int(in_domain_data_num),
                                                    "acc": round(in_domain_ins_acc, 3)
                                                    }
            # add sub category
            for cat_name, cat_results in in_domain_cat_results.items():
                printable_results[cat_name] = {"num": int(cat_results['num_example']),
                                            "acc": round(cat_results['acc'], 3)
                                            }
            
        # table.append(["-----------------------------", "-----", "----"])
        all_ins_acc = calculate_ins_level_acc(evaluation_result)
        printable_results['Overall'] = {"num": sum([cat_results['num_example'] for cat_results in evaluation_result.values()]),
                                        "acc": round(all_ins_acc, 3)
                                        }

        accel.print(printable_results)
        return 
    
    def evaluate_mathvista(self, model, accel):
        # MathVista Evaluation
        pred_answers = [{'pid': inputs['id'], 'image': inputs['id'], 'response': answer, 
                         'question_type': inputs['question_type'], 'answer_type': inputs['answer_type'], 'metadata': inputs['metadata'], 
                         'choices': inputs['choices'], 'query': inputs['question'], 'precision': inputs['precision'],} for inputs, answer in zip(self.inputs, self.gen_answers)]
        predictions = {pred['pid']: pred for pred in pred_answers}
        pred_pth = os.path.join(self.save_dir, f"{model}_mathvista_results.json")
        json.dump(predictions, open(pred_pth, "w"))
        
        accel.print(f"Finished evaluating MathVista. Evaluate the result file saved to {pred_pth}.")
        return 
    
    def evaluate_ai2d(self, model, accel):
        # AI2D Evaluation
        pred_answers = [{'question_id': inputs['id'], 'answer': answer, 'gt': inputs['gt']} for inputs, answer in zip(self.inputs, self.gen_answers)]
        pred_pth = os.path.join(self.save_dir, f"{model}_ai2d_results.json")
        json.dump(pred_answers, open(pred_pth, "w"))

        # Compute accuracy
        pattern = re.compile(r'[A-Z]')
        results = [(char_to_int(pattern.findall(answer)[0]) == inputs['gt']) for inputs, answer in zip(self.inputs, self.gen_answers)]

        accel.print(f"AI2D Accuracy: {np.mean(results)*100} %")
        return np.mean(results)*100
    
    def evaluate_hallusionbench(self, model, accel):
        # HallusionBench Evaluation
        pred_answers = [{'answer': '1' if answer.lower().find('yes') != -1 else '0', 'question': inputs['question'], 'gt': inputs['gt']} for inputs, answer in zip(self.inputs, self.gen_answers)]
        pred_pth = os.path.join(self.save_dir, f"{model}_hallusionbench_results.json")
        json.dump(pred_answers, open(pred_pth, "w"))

        # Compute accuracy
        results = [(answer['answer'] == answer['gt']) for answer in pred_answers]
        accel.print(f"HallusionBench Accuracy: {np.mean(results)*100} %")
        return np.mean(results)*100
    
    def evaluate_chartqa(self, model, accel):
        # ChartQA Evaluation
        # post processing
        processed_answers = []
        for x in self.gen_answers:
            if any(i.isdigit() for i in x):
                processed_answers.append(x.split(" ")[0])
            else:
                processed_answers.append(x)
        pred_answers = [{'answer': answer, 'question': inputs['question'], 'annotation': inputs['gt']} for inputs, answer in zip(self.inputs, processed_answers)]
        pred_pth = os.path.join(self.save_dir, f"{model}_chartqa_results.json")
        json.dump(pred_answers, open(pred_pth, "w"))

        # Compute accuracy
        acc = evaluate_relaxed_accuracy(pred_answers)
        accel.print(f"ChartQA Accuracy: {acc*100}%")
        return acc
    
    def evaluate_seed(self, model, accel):
        # SEED Evaluation
        pred_answers = [{'answer': answer, 'question': inputs['question'], 'question_id': inputs['id'], 'gt': inputs['gt'], 'question_type': inputs['question_type']} for inputs, answer in zip(self.inputs, self.gen_answers)]
        pred_pth = os.path.join(self.save_dir, f"{model}_seed_results.json")
        json.dump(pred_answers, open(pred_pth, "w"))

        # Compute accuracy
        results = [(answer['answer'] == answer['gt']) for answer in pred_answers]
        accel.print (f"SEED Accuracy: {np.mean(results)*100} %")

        # Per question type accuracy
        for k, v in SEED_TYPES.items():
            sub_results = []
            for pred in pred_answers:
                if pred['question_type'] == k:
                    sub_results.append(pred['answer'] == pred['gt'])
            accel.print (f"{v}: {np.mean(sub_results)*100} %")
        
        return np.mean(results)*100
    
    def evaluate_llava(self, model, accel):
        # LLaVA-in-the-Wild Evaluation
        pred_answers = [{'question_id': inputs['id'],  'prompt': inputs['question'], 'text': answer, "answer_id": shortuuid.uuid()} for inputs, answer in zip(self.inputs, self.gen_answers)]
        sorted_answers = sorted(pred_answers, key=lambda x: x['question_id'])
        pred_pth = os.path.join(self.save_dir, f'{model}_llava_results.jsonl')
        ans_file = open(pred_pth, "w")
        for pred in sorted_answers:
            ans_file.write(json.dumps(pred) + "\n")
            ans_file.flush()
        ans_file.close()

        accel.print(f"Finished evaluating LLaVA-in-the-wild. Evaluate the result file saved to {pred_pth}.")
        return 
    
    def evaluate_blink(self, model, accel):
        # BLINK Evaluation
        # TODO
        return 
    
    def evaluate_mathverse(self, model, accel):
        # Mathverse Evaluation
        pred_answers = [{'sample_index' : inputs['id'], 'problem_index' : inputs['problem_index'], 'problem_version' : inputs['problem_version'],
                        'question' : inputs['origin_question'], 'answer' : inputs['gt'],
                        'question_type': inputs['question_type'], 'question_type': inputs['question_type'], 
                        'metadata': inputs['metadata'], 'query_wo': inputs['question'], 'query_cot' : inputs['query_cot'], 'model_answer' : answer} for inputs, answer in zip(self.inputs, self.gen_answers)]

        # answers = [item for item in pred_answers if item['problem_version'] != 'Text_Only']
        # text_only_answers = [item for item in pred_answers if item['problem_version'] == 'Text_Only']
        
        pred_pth = os.path.join(self.save_dir, f'{model}_mathverse_results.json')
        json.dump(pred_answers, open(pred_pth, "w"))
        pred_pth = os.path.join(self.save_dir, f'{model}_mathverse_scores.json')
        eval_mathverse(self.save_dir, pred_answers,f'{model}_mathverse_extracts.json', f'{model}_mathverse_scores.json')
        accel.print(f"Finished evaluating MathVerse. Evaluate the result file saved to {pred_pth}.")
        # TODO
        return 

    def evaluate_mmstar(self, model, accel):
        pred_answers = [{'question': inputs['question'],
                        'answer': inputs['answer'],
                        'category': inputs['category'],
                        'l2_category': inputs['l2_category'],
                        # 'bench': inputs['bench'],
                        'prediction' : answer} for inputs, answer in zip(self.inputs, self.gen_answers)]
        
        pred_pth = os.path.join(self.save_dir, f'{model}_mmstar_results.json')
        json.dump(pred_answers, open(pred_pth, "w"))
        
        df = pd.DataFrame(pred_answers)
                        
        eval_mmstar(df, self.save_dir, f'{model}_mmstar_scores.json')
        pred_pth = os.path.join(self.save_dir, f'{model}_mmstar_scores.json')
        accel.print(f"Finished evaluating MMStar. Evaluate the result file saved to {pred_pth}.")
