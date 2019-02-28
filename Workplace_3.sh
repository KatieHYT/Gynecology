for i in {1,2,3,4,5}; do python3 0A_multi_task_concat_FHB.py --fold=$i --gpu_id='3' --struc='mimic_previous_FHB' --weight_balance=1  --epoch=1500 --layers=10 --summary_file='Exp0_A' --cv_path='/home/katieyth/gynecology/data/5_fold_CMU_rs_13/';done

