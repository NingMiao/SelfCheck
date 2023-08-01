#Running
python ./script/run.py --dataset mathqa --start_id 1 --end_id 1320 --generation_repeat_time 10

#Evaluate prediction accuracy
python ./script/evaluate-prediction.py --dataset mathqa --start_id 1 --end_id 1320 --generation_repeat_time 10 --get_weight_choice 1

#Evaluate checking accuracy
python ./script/evaluate-checking.py --dataset mathqa --start_id 1 --end_id 1320 --generation_repeat_time 10