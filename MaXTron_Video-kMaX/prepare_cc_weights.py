import copy
import sys
import torch

pretrained_ckpt_path = sys.argv[1]
new_ckpt_path = sys.argv[2]

pretrained_weight = torch.load(pretrained_ckpt_path, map_location='cpu')
print(pretrained_weight.keys())

new_pretrained_weight = copy.deepcopy(pretrained_weight)

old_dict = pretrained_weight['model']
new_dict = copy.deepcopy(old_dict)

for module_name in old_dict.keys():
    if ("_class_embedding_projection" in module_name or "_mask_embedding_projection" in module_name or "_transformer_mask_head" in module_name or "_transformer_class_head" in module_name or "_pixel_space_mask_batch_norm" in module_name) and ('_kmax_transformer_layers' not in module_name):
        new_module_name = module_name.replace('sem_seg_head.predictor', 'cross_clip_tracking_module')
        print(module_name, new_module_name)
        new_dict[new_module_name] = old_dict[module_name]

new_pretrained_weight['model'] = new_dict
torch.save(new_pretrained_weight, new_ckpt_path)