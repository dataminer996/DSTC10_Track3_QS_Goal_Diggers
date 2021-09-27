
train feature:
cd task4/generate/generagefeature
python3  track3_retrival_train.py  gen_train.bin generateresult retrival_train.bin

cp retrival_train.bin  ../../retrieval/oscar_retri_v1/data/train.bin

traincmd:
cd task4/retrival/oscar_retri_v1
triancmd:
python3 lasttrain.py 5e-5  15 2.0

pred feature:
cd task4/generate/generagefeature
python3 track3_retrival_pred.py gen_dev.bin data/simmc2_dials_dstc10_devtest_retrieval_candidates.json  data/simmc2_dials_dstc10_devtest.json retrival_dev.bin


predcmd:
python3 last_pred.py modelgood find


python3 ./format_task4_retrieval_0923.py  --dir-bin oscar_retri_v1/0918/find_0822_lowercase_lrnew_5e-05batch3weight2.0gpu6_epoch_15/checkpoint-14-10575/resultfromsys.bin  --split-path ../generate/generagefeature/data/simmc2_dials_dstc10_devtest.json  --save-path pred-subtask-4-retrieval.json 



