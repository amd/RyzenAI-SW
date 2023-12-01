#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
""" ONNX-runtime validation script

This script was created to verify accuracy and performance of exported ONNX
models running with the onnxruntime. It utilizes the PyTorch dataloader/processing
pipeline for a fair comparison against the originals.

Copyright 2020 Ross Wightman
"""
import argparse
import numpy as np
import onnxruntime
from timm.data import create_loader, resolve_data_config, create_dataset
from timm.models import create_model
from timm.utils import AverageMeter
import time

parser = argparse.ArgumentParser(description='ONNX Validation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--onnx-input',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to onnx model/weights file')
parser.add_argument('--onnx-float',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to onnx model/weights file')
parser.add_argument('--onnx-quant',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to onnx model/weights file')
parser.add_argument('--onnx-output-opt',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to output optimized onnx graph')
parser.add_argument('--profile',
                    action='store_true',
                    default=False,
                    help='Enable profiler output.')
parser.add_argument('--model',
                    '-m',
                    metavar='NAME',
                    default='resnet50.tv_in1k',
                    help='model architecture (default: resnet50.tv_in1k)')
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--img-size',
                    default=None,
                    type=int,
                    metavar='N',
                    help='Input image dimension, uses model default if empty')
parser.add_argument('--mean',
                    type=float,
                    nargs='+',
                    default=None,
                    metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std',
                    type=float,
                    nargs='+',
                    default=None,
                    metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--crop-pct',
                    type=float,
                    default=None,
                    metavar='PCT',
                    help='Override default crop pct of 0.875')
parser.add_argument('--interpolation',
                    default='',
                    type=str,
                    metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--gpu',
                    action='store_true',
                    default=False,
                    help='Enable profiler output.')


def main():
    args = parser.parse_args()
    args.gpu_id = 0

    # Set graph optimization level
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.profile:
        sess_options.enable_profiling = True
    if args.onnx_output_opt:
        sess_options.optimized_model_filepath = args.onnx_output_opt
    if args.gpu:
        providers = ['CUDAExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    if args.onnx_input:
        session = onnxruntime.InferenceSession(args.onnx_input,
                                               sess_options,
                                               providers=providers)
        input_model_path = args.onnx_input
        model_name = args.model

        model = create_model(
            model_name,
            pretrained=False,
        )
        data_config = resolve_data_config(vars(args),
                                          model=model,
                                          use_test_size=True)
        print("data config:")
        print(data_config)
        loader = create_loader(create_dataset('', args.data),
                               input_size=data_config['input_size'],
                               batch_size=args.batch_size,
                               use_prefetcher=False,
                               interpolation=data_config['interpolation'],
                               mean=data_config['mean'],
                               std=data_config['std'],
                               num_workers=args.workers,
                               crop_pct=data_config['crop_pct'])

        input_name = session.get_inputs()[0].name

        batch_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        for i, (input, target) in enumerate(loader):
            # run the net and return prediction
            output = session.run([], {input_name: input.data.numpy()})
            output = output[0]

            # measure accuracy and record loss
            prec1, prec5 = accuracy_np(output, target.numpy())
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    f'Test: [{i}/{len(loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {input.size(0) / batch_time.avg:.3f}/s, '
                    f'{100 * batch_time.avg / input.size(0):.3f} ms/sample) \t'
                    f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')

        print(
            f' * Prec@1 {top1.avg:.3f} ({100-top1.avg:.3f}) Prec@5 {top5.avg:.3f} ({100.-top5.avg:.3f})'
        )
    elif args.onnx_float and args.onnx_quant:
        session_f = onnxruntime.InferenceSession(args.onnx_float,
                                                 sess_options,
                                                 providers=providers)
        model_name = args.model

        model = create_model(
            model_name,
            pretrained=False,
        )
        data_config = resolve_data_config(vars(args),
                                          model=model,
                                          use_test_size=True)
        print("data config:")
        print(data_config)
        loader = create_loader(create_dataset('', args.data),
                               input_size=data_config['input_size'],
                               batch_size=args.batch_size,
                               use_prefetcher=False,
                               interpolation=data_config['interpolation'],
                               mean=data_config['mean'],
                               std=data_config['std'],
                               num_workers=args.workers,
                               crop_pct=data_config['crop_pct'])

        input_name = session_f.get_inputs()[0].name

        batch_time = AverageMeter()
        f_top1 = AverageMeter()
        f_top5 = AverageMeter()
        end = time.time()
        print("Evaluate float model...")
        for i, (input, target) in enumerate(loader):
            # run the net and return prediction
            output = session_f.run([], {input_name: input.data.numpy()})
            output = output[0]

            # measure accuracy and record loss
            prec1, prec5 = accuracy_np(output, target.numpy())
            f_top1.update(prec1.item(), input.size(0))
            f_top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    f'Test: [{i}/{len(loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {input.size(0) / batch_time.avg:.3f}/s, '
                    f'{100 * batch_time.avg / input.size(0):.3f} ms/sample) \t'
                    f'Prec@1 {f_top1.val:.3f} ({f_top1.avg:.3f})\t'
                    f'Prec@5 {f_top5.val:.3f} ({f_top5.avg:.3f})')

        f_top1 = format(f_top1.avg, '.2f')
        f_top5 = format(f_top5.avg, '.2f')
        session_q = onnxruntime.InferenceSession(args.onnx_quant,
                                                 sess_options,
                                                 providers=providers)
        q_top1 = AverageMeter()
        q_top5 = AverageMeter()

        print("Evaluate quantized model...")
        for i, (input, target) in enumerate(loader):
            # run the net and return prediction
            output = session_q.run([], {input_name: input.data.numpy()})
            output = output[0]

            # measure accuracy and record loss
            prec1, prec5 = accuracy_np(output, target.numpy())
            q_top1.update(prec1.item(), input.size(0))
            q_top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    f'Test: [{i}/{len(loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {input.size(0) / batch_time.avg:.3f}/s, '
                    f'{100 * batch_time.avg / input.size(0):.3f} ms/sample) \t'
                    f'Prec@1 {q_top1.val:.3f} ({q_top1.avg:.3f})\t'
                    f'Prec@5 {q_top5.val:.3f} ({q_top5.avg:.3f})')

        q_top1 = format(q_top1.avg, '.2f')
        q_top5 = format(q_top5.avg, '.2f')
        import os
        f_size = format(os.path.getsize(args.onnx_float) / (1024 * 1024), '.2f')
        q_size = format(os.path.getsize(args.onnx_quant) / (1024 * 1024), '.2f')
        """
        --------------------------------------------------------
        |             | float model    | quantized model |
        --------------------------------------------------------
        | ****        | ****           | ****             |
        --------------------------------------------------------
        | Model Size  | ****           | ****             |
        --------------------------------------------------------
        """
        from rich.console import Console
        from rich.table import Table
        console = Console()

        table = Table()
        table.add_column('')
        table.add_column('Float Model')
        table.add_column('Quantized Model', style='bold green1')

        table.add_row("Model", args.onnx_float, args.onnx_quant)
        table.add_row("Model Size", str(f_size) + ' MB', str(q_size) + ' MB')
        table.add_row("Prec@1", str(f_top1) + ' %', str(q_top1) + ' %')
        table.add_row("Prec@5", str(f_top5) + ' %', str(q_top5) + ' %')

        console.print(table)

    else:
        print(
            "Please specify both model-float and model-quant or model-input for evaluation."
        )


def accuracy_np(output, target):
    max_indices = np.argsort(output, axis=1)[:, ::-1]
    top5 = 100 * np.equal(max_indices[:, :5],
                          target[:, np.newaxis]).sum(axis=1).mean()
    top1 = 100 * np.equal(max_indices[:, 0], target).mean()
    return top1, top5


if __name__ == '__main__':
    main()
