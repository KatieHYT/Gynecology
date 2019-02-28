# for i in {1,2,3,4,5}; do python3 0C_multi_task.py --fold=$i --gpu_id='6' --struc='deeper' --weight_balance=1  --epoch=1500 --layers=10 ;done
# for i in {1,2,3,4,5}; do python3 0C_multi_task.py --fold=$i --gpu_id='6' --struc='deeper' --weight_balance=1  --epoch=1500 --layers=5 ;done



# for i in {1,2,3,4,5}; do python3 0B_multi_task_concat.py --fold=$i --gpu_id='6' --struc='man_concat' --weight_balance=1  --epoch=1500 --layers=10 ;done
# for i in {1,2,3,4,5}; do python3 0B_multi_task_concat.py --fold=$i --gpu_id='6' --struc='man_concat' --weight_balance=1  --epoch=1500 --layers=5 ;done



# for i in {1,2,3,4,5}; do python3 0A_multi_task_concat_FHB.py --fold=$i --gpu_id='6' --struc='mimic_previous_FHB' --weight_balance=1  --epoch=1500 --layers=10 ;done
# for i in {1,2,3,4,5}; do python3 0A_multi_task_concat_FHB.py --fold=$i --gpu_id='6' --struc='mimic_previous_FHB' --weight_balance=1  --epoch=1500 --layers=5 ;done


for i in {1,2,3,4,5}; do python3 0D_multi_task_FHB.py --fold=$i --gpu_id='6' --struc='multi_task_FHB' --weight_balance=1  --epoch=1500 --layers=10 --summary_file='Exp0_D' --cv_path='/home/katieyth/gynecology/data/5_fold_CMU_rs_13/';done

