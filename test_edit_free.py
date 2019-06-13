### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


# giving the edit label and where does each part come from



import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

def merge_image(no_bg_tensor, bg_tensor, label):
    assert no_bg_tensor.dim() == 3
    mask = (label == 0).type(torch.FloatTensor)
    mask_f = (label != 0).type(torch.FloatTensor)
    return no_bg_tensor.type(torch.FloatTensor)*mask_f + bg_tensor * mask

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx
    
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1 
    

    img_path = data['path']
    print('process image... %s' % img_path)

    if opt.test_type == 'encode':
        print(web_dir+"/encode_tensor")
        if os.path.exists(web_dir+"/encode_tensor") == False:
            os.mkdir(web_dir+"/encode_tensor")
        generated = model.inference_encode(data['path'], data['bg_styleimage'], data['label'], data['mask'])

    if opt.test_type == 'generate':
        generated = model.inference_generate(data['path'], data['bg_styleimage'], data['label'], data['mask'])

    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                   ('style_image', util.tensor2im(data['bg_styleimage'][0])),
                   ('reconstruct_style_image', util.tensor2im(generated.data[0])),
                   ])
            
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()


# python test_twomask.py --name helen_mask/helen_mask_vae_debug36_6 --no_instance --dataroot ./datasets/helen_align/ --resize_or_crop none --label_nc 11 --n_downsample_global 2 --longSize 256 --norm batch --mask_model --debug_mask_part --phase test4 --phase2 test5 --return_bg