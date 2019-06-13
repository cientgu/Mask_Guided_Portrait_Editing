### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
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
    
    # if opt.engine:
    #     generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    # elif opt.onnx:
    #     generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    # else:        
    #     generated = model.inference(data['label'], data['inst'], data['image'], type="none")

    # visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
    #                ('real_image', util.tensor2im(data['image'][0])),
    #                ('synthesized_image', util.tensor2im(generated.data[0]))
    #                ])

    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:        
        if opt.debug_mask_part == True:
            generated,generated2,generated3 = model.inference(data['bg_contentimage'],data['label2'], data['mask2'], data['bg_styleimage'], data['label'], data['mask'])

            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('input_label2', util.tensor2label(data['label2'][0], opt.label_nc)),
                           ('style_image', util.tensor2im(data['bg_styleimage'][0])),
                           ('content_image', util.tensor2im(data['bg_contentimage'][0])),
                           ('reconstruct_style_image', util.tensor2im(generated.data[0])),
                           ('reconstruct_content_image', util.tensor2im(generated2.data[0])),
                           ('crop_mouth', util.tensor2im(generated3.data[0]))
                           ])
            # else:
            #     visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
            #                    ('input_label2', util.tensor2label(data['label2'][0], opt.label_nc)),
            #                    ('real_image', util.tensor2im(data['image'][0])),
            #                    ('synthesized_mask_image', util.tensor2im(generated.data[0])),
            #                    ('synthesized_mask2_image', util.tensor2im(generated2.data[0]))
            #                    ])
        else:
            generated = model.inference(data['label'], data['inst'], data['image'], data['mask'])
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('real_image', util.tensor2im(data['image'][0])),
                           ('synthesized_mask_image', util.tensor2im(generated.data[0]))
                           ])





    
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()


# python test_twomask.py --name helen_mask/helen_mask_vae_debug36_6 --no_instance --dataroot ./datasets/helen_align/ --resize_or_crop none --label_nc 11 --n_downsample_global 2 --longSize 256 --norm batch --mask_model --debug_mask_part --phase test4 --phase2 test5 --return_bg