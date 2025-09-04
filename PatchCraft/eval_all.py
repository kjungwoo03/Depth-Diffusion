'''
python eval_all.py --model_path ./weights/{}.pth --detect_method {CNNSpot,Gram,Fusing,FreDect,LGrad,LNP,DIRE}  --noise_type {blur,jpg,resize}
'''

import os
import csv

from numpy import mean
import torch

from validate import validate
from options import TestOptions
from eval_config import *
from PIL import ImageFile
from util import  set_random_seed
from networks import Net as RPTC
from networks import initWeights

ImageFile.LOAD_TRUNCATED_IMAGES = True


# 固定随机种子
set_random_seed()
# Running tests


opt = TestOptions().parse(print_options=True) #获取参数类



model_name = os.path.basename(opt.model_path).replace('.pth', '')

mkdir(opt.results_dir)

rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)


    model = RPTC()
    model.apply(initWeights)
    state_dict = torch.load(opt.model_path, map_location='cpu')

    try:
        model.load_state_dict(state_dict['netC'],strict=True)
    except:
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['netC'].items()})

    model.cuda()
    model.eval()


    acc, ap, _, _, _, _ = validate(model, opt)
    rows.append([val, acc, ap])
    print("({}) acc: {}; ap: {}".format(val, acc, ap))
mean_acc = []
mean_ap = []
for i in range(len(rows)-2):
    mean_acc.append(rows[i+2][1])
    mean_ap.append(rows[i+2][2])
mean_acc = mean(mean_acc)
mean_ap = mean(mean_ap)
print("({}) acc: {}; ap: {}".format('AVG',mean_acc, mean_ap))


# 结果文件
csv_name = opt.results_dir + '/{}.csv'.format(opt.noise_type)
with open(csv_name, 'a+') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
