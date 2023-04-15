#!/bin/bash

python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
sed -i'.py' 's/MCAR/MAR/g' synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
sed -i'.py' 's/MAR/MNAR/g' synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py

sed -i'.py' 's/_missing_0.1/_missing_0.2/g' synthesizer/syn_bank.py
sed -i'.py' 's/MNAR/MCAR/g' synthesizer/syn_bank.py

python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
sed -i'.py' 's/MCAR/MAR/g' synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
sed -i'.py' 's/MAR/MNAR/g' synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py

sed -i'.py' 's/_missing_0.2/_missing_0.3/g' synthesizer/syn_bank.py
sed -i'.py' 's/MNAR/MCAR/g' synthesizer/syn_bank.py

python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
sed -i'.py' 's/MCAR/MAR/g' synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py
sed -i'.py' 's/MAR/MNAR/g' synthesizer/syn_bank.py
python synthesizer/syn_bank.py
python synthesizer/syn_bank.py