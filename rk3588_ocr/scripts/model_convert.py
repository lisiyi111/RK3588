import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
import argparse

target_platform_list = ['rk3566', 'rk3568', 'rk3588', 'rv1103', 'rv1106', 'rk3562','rk3576']


def parse_args():
    parser = argparse.ArgumentParser(description="st infer")
    parser.add_argument("-i", "--input_model", type=str, help="the input onnx model path")
    parser.add_argument("-o", "--output_model", type=str, help="the output rknn model path")
    parser.add_argument("--do_quantization", action='store_true', help="quantization")
    parser.add_argument("--quant_dataset", type=str, default=None, help="the quantization dataset path:txt")
    parser.add_argument("--target_platform", type=str, default=None, help="target_platform")
    parser.add_argument("--mean_values", nargs='+', type=float, default=[0, 0, 0], help="mean_values")
    parser.add_argument("--std_values", nargs='+', type=float, default=[255, 255, 255], help="std_values")
    parser.add_argument("--not_normal", action='store_true', help="not use Normalization")
    parser.add_argument("--auto_prune", action='store_true', help="auto_prune")
    parser.add_argument("--do_analysis", action='store_true', help="acc analysis")

    args = parser.parse_args()
    return args


def get_analysis_data(txt_path):
    with open(txt_path, 'r') as fr:
        analysis_data = fr.readlines()[0].strip()
    return analysis_data


def main():
    args = parse_args()
    input_model = args.input_model
    output_model = args.output_model
    do_quantization = args.do_quantization
    quant_dataset = args.quant_dataset
    std_values = args.std_values
    mean_values = args.mean_values
    target_platform = args.target_platform
    not_normal = args.not_normal
    auto_prune = args.auto_prune
    do_analysis = args.do_analysis

    if target_platform is not None:
        if target_platform not in target_platform_list:
            print("[ERROR] Not have {}".format(target_platform))
            exit(-1)

    # Create RKNN object
    rknn = RKNN(verbose=True)
    # pre-process config
    print('--> Config model')
    if not_normal:
        rknn.config(target_platform=target_platform, model_pruning=auto_prune)
    else:
        rknn.config(mean_values=[mean_values],
                    std_values=[std_values],
                    target_platform=target_platform,
                    model_pruning=auto_prune)

    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=input_model)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')

    ret = rknn.build(do_quantization=do_quantization, dataset=quant_dataset)

    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_model)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    if do_analysis:
        if not quant_dataset:
            print("Do analysis must set quant_dataset")
            exit(-1)
        # Init runtime environment
        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        ## 连板分析
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('done')

        img_file = get_analysis_data(quant_dataset)
        ret = rknn.accuracy_analysis(inputs=[img_file])
        if ret != 0:
            print('Accuracy analysis failed!')
            exit(ret)
        print('done')

    rknn.release()

if __name__ == '__main__':
    main()
