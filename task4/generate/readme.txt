data and model download url:
and download the simc2 data to 

data and model copy:
cd task4
unzip task4.tar to task4 
cp task4/generate/data/oscargen_noobj_newdata/data generate/oscargen_noobj_newdata/ -fr
cp task4/generate/model/oscargen_noobj_newdata/*  generate/oscargen_noobj_newdata/ -fr
cp task4/generate/model/generagefeature/*  generate/generagefeature/ -fr
 

train feature:
python3 track3_gen_train.py 0 9000
cp gentrain.bin ../oscargen_noobj_newdata/data/train.bin

train cmd:
cd task4/generate/oscargen_noobj_newdata
python3 lasttrin.py 17e-5  240


eval feature:
cd task4/generate/generagefeature 
python3 track3_gen_pred.py data/simmc2_dials_dstc10_devtest.json
cp gendev.bin  ../oscargen_noobj_newdata/data/dev.bin


eval cmdline:
cd task4/generate/oscargen_noobj_newdata
python3 autotraineval_strdir.py  modelgood  find
cd task4/generate
python3  format_task4_generation.py  --generation-pred-txt  oscargen_noobj_newdata/modelgood/find_0822lrnew_0.00017batch18weight3gpu1_epoch_242/checkpoint-241-42592/resulefile.txt --split-path  generagefeature/data/simmc2_dials_dstc10_devtest.json     --save-path result/subtask-4-generation.json



