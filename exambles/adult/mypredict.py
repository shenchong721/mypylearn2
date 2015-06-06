#!/usr/bin/env python2
# coding: utf-8

from pylearn2.scripts import predict_csv


"""
See module-level docstring for a description of the script.
"""
parser = predict_csv.make_argument_parser()
args = parser.parse_args()
args.has_row_label=True
ret = predict_csv.predict(args.model_filename, args.test_filename, args.output_filename,
    args.prediction_type, args.output_type,
    args.has_headers, args.has_row_label, args.delimiter)
if not ret:
    sys.exit(-1)


