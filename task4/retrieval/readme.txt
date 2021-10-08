you need to do the genearate feature first
1.data and model download
cd task4
download gs://tangliang-commit/public/task4  to task4
cp task4/retrival/data/* task4/retrival/oscar_retri_v1/ -fr
cp task4/retrival/model/* task4/retrival/oscar_retri_v1/ -fr


2. train feature:
docker file: gs://tangliang-commit/public/featuredocker.tar
cd task4/generate/generagefeature
python3  track3_retrival_train.py  gen_train.bin generateresult retrival_train.bin
need to run in docker file  gs://tangliang-commit/


cp retrival_train.bin  ../../retrieval/oscar_retri_v1/data/train.bin

3. train command:
docker file: gs://tangliang-commit/public/oscarandfewshot.tar

cd task4/retrival/oscar_retri_v1 (need GPU)
python3 lasttrain.py 5e-5  15 2.0


4. pred feature:
docker file: gs://tangliang-commit/public/featuredocker.tar
cd task4/generate/generagefeature
python3 track3_retrival_pred.py gen_dev.bin data/simmc2_dials_dstc10_devtest_retrieval_candidates.json  data/simmc2_dials_dstc10_devtest.json retrival_dev.bin (for devtest)
python3 track3_retrival_pred_teststd.py gen_dev.bin data/simmc2_dials_dstc10_teststd_retrieval_candidates_public  data/simmc2_dials_dstc10_teststd_public.json retrival_dev.bin (for teststd)
cp retrival_dev.bin  ../../retrieval/oscar_retri_v1/data/dev.bin



5. predcmd:
docker file: gs://tangliang-commit/public/oscarandfewshot.tar

python3 last_pred.py modelgood find (need GPU)

(for devtest)
python3 ./format_task4_retrieval_0923.py  --dir-bin oscar_retri_v1/0918/find_0822_lowercase_lrnew_5e-05batch3weight2.0gpu6_epoch_15/checkpoint-14-10575/resultfromsys.bin  --split-path ../generate/generagefeature/data/simmc2_dials_dstc10_devtest.json  --save-path  ./result/dstc10-simmc-devtest-pred-subtask-4-retrieval.json

(for teststd)
python3 ./format_task4_retrieval_0923_teststd.py \
 --dir-bin oscar_retri_v1/0918/find_0822_lowercase_lrnew_5e-05batch3weight2.0gpu6_epoch_15/checkpoint-14-10575/resultfromsys.bin \
 --split-path ../generate/generagefeature/data/simmc2_dials_dstc10_teststd_retrieval_candidates_public.json \
 --save-path  ./result/dstc10-simmc-teststd-pred-subtask-4-retrieval.json

