you need to run the task1 prediction code first to pred the result

1. downlaod the simmc2 data
cd task4/generate/generagefeature
and download the https://github.com/facebookresearch/simmc2/tree/master/data   to  task4/generate/generagefeature/data
and unzip the simmc2_scene_images_dstc10_public_part2.zip simmc2_scene_images_dstc10_public_part1.zip
unzip simmc2_scene_jsons_dstc10_public.zip
make sure the dir name is data/simmc2_scene_images_dstc10_public_part1 data/simmc2_scene_images_dstc10_public_part2 data/public

2. download traindata and model:
cd task2
dowload task2.zip
unzip task2.zip
cp  task2/data/SSL-FEW-SHOT/*  SSL-FEW-SHOT/
cp  task2/model/SSL-FEW-SHOT/*  SSL-FEW-SHOT/
cp  task2/data/oscarv1_newdata/*  oscarv1_newdata/
cp  task2/model/oscarv1_newdata/*  oscarv1_newdata/

3. object model train command:
cd task2/SSL-FEW-SHOT
furniture:
python3 train.py 30 10 0.007 data/miniimagenet/split_furniture track3furniture_resize128 10  (need GPU) 
fashion:
python3 train.py 20 17 0.007 data/miniimagenet/split_fashion track3fashion_resize128 90  (need GPU 32G memory)  


4. object id model train 
cd task2/oscarv1_newdata
python3 train.py  5.0e-5 19 3.7


5. create object type pred features:
cd task4/generate/generagefeature
mkdir devtestobjimage
python3 track3_saveobj_rep.py  data/simmc2_dials_dstc10_devtest.json  devtestobjimage
python3  rw_devtestimagecsv.py devtestobjimage val.csv
/bin/cp val.csv  task2/SSL-FEW-SHOT/data/predcsv/0927_fashion/.
/bin/cp val.csv  task2/SSL-FEW-SHOT/data/predcsv/0927_furniture/.
/bin/cp  devtestobjimage/*  task2/SSL-FEW-SHOT/data/image/imagetraindev0927/.


6. object type pred cmd
cd task2/SSL-FEW-SHOT
python3 test.py 60 17 track3fashion_resize128/track3_lr0.0040_20_17way_maxepoch90 data/predcsv/0927_fashion data/image/imagetraindev0927
python3 test.py 30 10 track3fashion_resize128/track3_lr0.0040_20_17way_maxepoch90 data/predcsv/0927_fashion data/image/imagetraindev0927


python3 readresult.py  data/predcsv/0927_fashion track3fashion_resize128   fashion_result.txt
python3 readresult.py  data/predcsv/0927_furniture track3furniture_resize128  furniture_result.txt
cp fashion_result.txt  task4/generate/generagefeature/data/
cp furniture_result.txt task4/generate/generagefeature/data/


7. object id prediction feature
cd task4/generate/generagefeature
python3 json_for_from_sys_all.py data/simmc2_dials_dstc10_devtest.json  ../../../task1/result/devtest_from_system_embedding.json 
python3 json_for_from_sys.py data/simmc2_dials_dstc10_devtest.json  ../../../task1/result/devtest_from_system_embedding.json
python3 track3_object_pred.py  test_s_object_pred.json   task1/result/devtest_objects_num_embedding.json
python3 from_system_bin.py test_s_object_pred.json  object_pred.bin object_pred_step1.bin
python3 changefeature_forpred.py object_pred_step1.bin  object_pred_last.bin
cp object_pred_last.bin  task2/oscarv1_newdata/data/dev.bin


8. object id pred
python3 last_pred.py modelgood  
python3 getsingleresult.py modelgood/find_0822_lowercase_lrnew_5e-05batch3weight3.7gpu6_epoch_20/checkpoint-19-11580/resultfromsys.bin ../../task1/result/devtest_objects_num_embedding.json 
sh run_step2_predict.sh
