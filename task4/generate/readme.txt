1. downlaod the simmc2 data
cd task4/generate/generagefeature
and download the https://github.com/facebookresearch/simmc2/tree/master/data   to  task4/generate/generagefeature/data
and unzip the simmc2_scene_images_dstc10_public_part2.zip simmc2_scene_images_dstc10_public_part1.zip 
unzip simmc2_scene_jsons_dstc10_public.zip 
make sure the dir name is data/simmc2_scene_images_dstc10_public_part1 data/simmc2_scene_images_dstc10_public_part2 data/public  

2. data and model copy:
cd task4
#download https://drive.google.com/drive/folders/1ILTFnaRTTcGWAzYXJnt_3QmeiVgE501T?usp=sharing as task4.zip
download gs://tangliang-commit/public/task4  to task4
cp task4/generate/data/oscargen_noobj_newdata/data generate/oscargen_noobj_newdata/ -fr
cp task4/generate/model/oscargen_noobj_newdata/*  generate/oscargen_noobj_newdata/ -fr
cp task4/generate/model/generagefeature/*  generate/generagefeature/ -fr
 

3. train feature:
docker file: gs://tangliang-commit/public/featuredocker.tar
cd task4/generate/generagefeature
chmod +x run.sh
python3 track3_gen_train.py 0 9000  (need GPU)
cp gentrain.bin ../oscargen_noobj_newdata/data/train.bin

4. train cmd:
docker file: gs://tangliang-commit/public/oscarandfewshot.tar
cd task4/generate/oscargen_noobj_newdata  
python3 lasttrin.py 17e-5  240  (need GPU)


5. eval feature:
docker file: gs://tangliang-commit/public/featuredocker.tar
cd task4/generate/generagefeature 
(for devtest) python3 track3_gen_pred.py data/simmc2_dials_dstc10_devtest.json (need GPU)
(for teststd) python3 track3_gen_pred_teststd.py data/simmc2_dials_dstc10_teststd_public.json (need GPU)
cp gen_dev.bin  ../oscargen_noobj_newdata/data/dev.bin


6. eval cmdline:
docker file: gs://tangliang-commit/public/oscarandfewshot.tar
cd task4/generate/oscargen_noobj_newdata
python3 pred_strdir.py  modelgood  find  (need GPU)
cd task4/generate
(for devtest)
python3  format_task4_generation.py  --generation-pred-txt  oscargen_noobj_newdata/modelgood/find_0822lrnew_0.00017batch18weight3gpu1_epoch_242/checkpoint-241-42592/resulefile.txt --split-path  generagefeature/data/simmc2_dials_dstc10_devtest.json     --save-path result/subtask-4-generation.json

(for teststd) python3 ./format_task4_generation.py \
 --generation-pred-txt  oscargen_noobj_newdata/modelgood/find_0822lrnew_0.00017batch18weight3gpu1_epoch_242/checkpoint-241-42592/resulefile.txt \
 --split-path   ../generate/generagefeature/data/simmc2_dials_dstc10_teststd_public \
 --save-path  ./result/dstc10-simmc-teststd-pred-subtask-4-generation.json



