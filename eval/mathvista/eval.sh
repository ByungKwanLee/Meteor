# python eval/mathvista/extract_answer.py \
# --output_dir /mnt/ssd/lbk-cvpr/dataset/eval_results \
# --output_file Meteor_mathvista_results.json 

python eval/mathvista/calculate_score.py \
--output_dir /mnt/ssd/lbk-cvpr/dataset/eval_results \
--output_file Meteor_mathvista_results_refixed.json  \
--score_file Meteor_mathvista_scores.json \
--gt_file /mnt/ssd/lbk-cvpr/dataset/MathVista/annot_testmini.json \