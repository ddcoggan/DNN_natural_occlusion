import os
import os.path as op
for model_dir in os.listdir('DNN/weights'):
    out_path = f'DNN/weights/{model_dir}/cfg.json'
    if not op.isfile(out_path):
        os.rename(f'DNN/training/configs/{op.basename(model_dir)}.json',
            out_path)

