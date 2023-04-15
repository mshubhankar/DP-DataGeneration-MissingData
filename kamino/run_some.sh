#!/bin/bash
#start with MCAR 0.1
python synthesizer/syn_bank.py
python synthesizer/syn_adult.py
python synthesizer/syn_national.py
python synthesizer/syn_br2000.py