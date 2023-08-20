
## counterexample
expname="counterexample"
model="inputs/model_data/counterexample.csv"
params="inputs/params/counterexample_params.csv"
argsin="--expname ${expname} --model ${model} --params ${params}"
python run_experiments.py $argsin --n_chunks 8 > tmp/run.sh
echo wait >> tmp/run.sh
bash tmp/run.sh
python combine_results.py $argsin
python analysis.py $argsin --csv


# Maternal health/
expname="maternal_A"
model="inputs/model_data/maternal_A.csv"
params="inputs/params/maternal_params.csv"
argsin="--expname ${expname} --model ${model} --params ${params}"
python run_experiments.py $argsin --n_chunks 8 > tmp/run.sh
echo wait >> tmp/run.sh
bash tmp/run.sh
python combine_results.py $argsin
python analysis.py $argsin --csv

expname="maternal_B"
model="inputs/model_data/maternal_B.csv"
params="inputs/params/maternal_params.csv"
argsin="--expname ${expname} --model ${model} --params ${params}"
python run_experiments.py $argsin --n_chunks 8 > tmp/run.sh
echo wait >> tmp/run.sh
bash tmp/run.sh
python combine_results.py $argsin
python analysis.py $argsin --csv

expname="maternal_C"
model="inputs/model_data/maternal_C.csv"
params="inputs/params/maternal_params.csv"
argsin="--expname ${expname} --model ${model} --params ${params}"
python run_experiments.py $argsin --n_chunks 8 > tmp/run.sh
echo wait >> tmp/run.sh
bash tmp/run.sh
python combine_results.py $argsin
python analysis.py $argsin --csv


# ## Diabetes App
expname="DBApp"
model="inputs/model_data/marketscan.csv"
params="inputs/params/fullobs_params.csv"
argsin="--expname ${expname} --model ${model} --params ${params}"
python run_experiments.py $argsin --n_chunks 8 > tmp/run.sh
echo wait >> tmp/run.sh
bash tmp/run.sh
python combine_results.py $argsin
python analysis.py $argsin --csv

## Banner
expname="banner"
model="inputs/model_data/marketscan.csv"
params="inputs/params/fullobs_params.csv"
argsin="--expname ${expname} --model ${model} --params ${params}"
python banner_plot.py $argsin --csv

