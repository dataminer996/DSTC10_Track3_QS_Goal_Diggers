you need to run the task1 prediction code first to pred the result

1. downlaod the simmc2 data
cd task4/generate/generagefeature
and download the https://github.com/facebookresearch/simmc2/tree/master/data   to  task4/generate/generagefeature/data
and unzip the simmc2_scene_images_dstc10_public_part2.zip simmc2_scene_images_dstc10_public_part1.zip
unzip simmc2_scene_jsons_dstc10_public.zip
make sure the dir name is data/simmc2_scene_images_dstc10_public_part1 data/simmc2_scene_images_dstc10_public_part2 data/public

2. download traindata and model:
cd task2
download gs://tangliang-commit/public/task2  to task2
cp  task2/data/SSL-FEW-SHOT/*  SSL-FEW-SHOT/
cp  task2/model/SSL-FEW-SHOT/*  SSL-FEW-SHOT/
cp  task2/data/oscarv1_newdata/*  oscarv1_newdata/
cp  task2/model/oscarv1_newdata/*  oscarv1_newdata/

3. object model train command:
docker file: gs://tangliang-commit/public/oscarandfewshot.tar 
cd task2/SSL-FEW-SHOT
furniture:
python3 train.py 30 10 0.007 data/miniimagenet/split_furniture track3furniture_resize128 10  (need GPU) 
fashion:
python3 train.py 20 17 0.007 data/miniimagenet/split_fashion track3fashion_resize128 90  (need GPU 32G memory)  


4. object id model train 
docker file: gs://tangliang-commit/public/oscarandfewshot.tar 
cd task2/oscarv1_newdata
python3 train.py  5.0e-5 19 3.7


5. create object type pred features:
docker file: gs://tangliang-commit/public/featuredocker.tar 
cd task4/generate/generagefeature
mkdir devtestobjimage
python3   track3_save_replace.py   data/simmc2_dials_dstc10_devtest.json  devtestobjimage
python3  rw_devtestimagecsv.py devtestobjimage val.csv
/bin/cp val.csv  task2/SSL-FEW-SHOT/data/predcsv/0927_fashion/.
/bin/cp val.csv  task2/SSL-FEW-SHOT/data/predcsv/0927_furniture/.
/bin/cp  devtestobjimage/*  task2/SSL-FEW-SHOT/data/image/imagetraindev0927/.


6. object type pred cmd
docker file: gs://tangliang-commit/public/oscarandfewshot.tar 
cd task2/SSL-FEW-SHOT
python3 test.py 60 17 track3fashion_resize128/track3_lr0.0030_20_17way_maxepoch90 data/predcsv/0927_fashion data/image/imagetraindev0927
python3 test.py 30 10 track3furniture_resize128/track3_lr0.007_30_10way_maxepoch10 data/predcsv/0927_furniture data/image/imagetraindev0927


python3 readresult.py  data/predcsv/0927_fashion track3fashion_resize128   fashion_result.txt
python3 readresult.py  data/predcsv/0927_furniture track3furniture_resize128  furniture_result.txt
cp fashion_result.txt  task4/generate/generagefeature/data/
cp furniture_result.txt task4/generate/generagefeature/data/


7. object id prediction feature
docker file: gs://tangliang-commit/public/featuredocker.tar 
cd task4/generate/generagefeature
(for devtest)
python3 json_for_from_sys_all.py data/simmc2_dials_dstc10_devtest.json  ../../../task1/result/devtest_from_system_embedding.json 
python3 json_for_from_sys.py data/simmc2_dials_dstc10_devtest.json  ../../../task1/result/devtest_from_system_embedding.json
python3 track3_object_pred.py  test_s_object_pred.json   task1/result/devtest_objects_num_embedding.json
python3 from_system_bin.py test_s_object_pred.json  object_pred.bin object_pred_step1.bin
python3 changefeature_forpred.py object_pred_step1.bin  object_pred_last.bin
cp object_pred_last.bin  task2/oscarv1_newdata/data/dev.bin

(for teststd)
python3 json_for_from_sys_std.py data/simmc2_dials_dstc10_teststd_public.json  ../../../task1/result/teststd_from_system_embedding.json
python3 track3_object_pred_teststd.py  test_s_object_pred.json   ../../../task1/result/teststd_objects_num_embedding.json
python3 from_system_bin.py test_s_object_pred.json  object_pred.bin object_pred_step1.bin
python3 changefeature_forpred.py object_pred_step1.bin  object_pred_last.bin
cp object_pred_last.bin  task2/oscarv1_newdata/data/dev.bin


8. object id pred
docker file: gs://tangliang-commit/public/oscarandfewshot.tar 
python3 last_pred.py modelgood  
python3 getsingleresult.py modelgood/find_0822_lowercase_lrnew_5e-05batch3weight3.7gpu6_epoch_20/checkpoint-19-11580/resultfromsys.bin ../../task1/result/devtest_objects_num_embedding.json 

(for devtest)
python3 ./format_task2.py \
 --step2-bin  oscarv1_newdata/modelgood/find_0822_lowercase_lrnew_5e-05batch3weight3.7gpu6_epoch_20/checkpoint-19-11580/resultfromsys.binupdate \
 --split-path ../task4/generate/generagefeature/data/simmc2_dials_dstc10_devtest.json \
 --save-path ./results/dstc10-simmc-devtest-pred-subtask-2.json

(for teststd)
python3  format_task4_generation.py  --generation-pred-txt  oscargen_noobj_newdata/modelgood/find_0822lrnew_0.00017batch18weight3gpu1_epoch_242/checkpoint-241-42592/resulefile.txt --split-path  generagefeature/data/simmc2_dials_dstc10_devtest.json     --save-path result/subtask-4-generation.json
n3 ./format_task2.py \
 --step2-bin  oscarv1_newdata/modelgood/find_0822_lowercase_lrnew_5e-05batch3weight3.7gpu6_epoch_20/checkpoint-19-11580/resultfromsys.binupdate \
 --split-path ../task4/generate/generagefeature/data/simmc2_dials_dstc10_teststd_public.json \
 --save-path ./results/dstc10-simmc-teststd-pred-subtask-2.json
