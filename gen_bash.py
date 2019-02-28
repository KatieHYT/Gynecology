import os
import sys
import itertools

gpu_id = sys.argv[1]
model_save = '/data/put_data/cmchang/gynecology/model/'                 # s
target = 'variability'          # y
summary_file = '/data/put_data/cmchang/gynecology/variability-summary-rs13-random-noise.csv'              # fn

length = [300]             # l
random_noise = [1, 0]  # rn
normalized = [1]     # nm
# l_2 = [1e-6, 1e-4]                # l2
weight_balance = [1]     # wb
random_state = [13] # rs

combs = list(itertools.product(length, normalized, weight_balance, random_state, random_noise))
for para in combs:
    note = os.path.join(target, ('l%s-nm%s-wb%s-rs%s-noise%s-g%s' % (*para, gpu_id)))
    model_save_noted = os.path.join(model_save, note)
    script = 'python3 train.py -s {0} -y {1} -g {2} -fn {3}'.format(model_save_noted, target, gpu_id, summary_file)
    script = script + (' -l %s -nm %s -wb %s -rs %s -rn %s' % (para))
    print(script)
