#!/bin/bash

futhark cuda --library $1.fut
python extract_cuda.py $1
