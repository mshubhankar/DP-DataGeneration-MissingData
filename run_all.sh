#!/bin/bash
python main.py #start with MCAR 0.1
sed -i 's/MCAR/MAR/g' config.py
python main.py
sed -i 's/MAR/MNAR/g' config.py
python main.py
sed -i 's/MNAR/MNARQ/g' config.py
python main.py

sed -i 's/0.1/0.2/g' config.py #reset to MCAR 0.2
sed -i 's/MNARQ/MCAR/g' config.py

python main.py
sed -i 's/MCAR/MAR/g' config.py
python main.py
sed -i 's/MAR/MNAR/g' config.py
python main.py
sed -i 's/MNAR/MNARQ/g' config.py
python main.py

