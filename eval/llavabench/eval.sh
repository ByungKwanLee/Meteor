python eval/llavabench/eval_gpt_review_bench.py \
    --question /mnt/ssd/lbk-cvpr/dataset/llava-bench-in-the-wild/questions.jsonl \
    --context /mnt/ssd/lbk-cvpr/dataset/llava-bench-in-the-wild/context.jsonl \
    --rule eval/llavabench/rule.json \
    --answer-list \
        /mnt/ssd/lbk-cvpr/dataset/llava-bench-in-the-wild/answers_gpt4.jsonl \
        /mnt/ssd/lbk-cvpr/dataset/eval_results/Meteor_llava_results.jsonl \
    --output \
        /mnt/ssd/lbk-cvpr/dataset/eval_results/reviews_meteor_llava_results_step3.jsonl

python eval/llavabench/summarize_gpt_review.py -f /mnt/ssd/lbk-cvpr/dataset/eval_results/reviews_meteor_llava_results_step3.jsonl
