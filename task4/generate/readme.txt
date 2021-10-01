1. downlaod the simmc2 data
cd task4/generate/generagefeature
and download the https://github.com/facebookresearch/simmc2/tree/master/data   to  task4/generate/generagefeature/data
and unzip the simmc2_scene_images_dstc10_public_part2.zip simmc2_scene_images_dstc10_public_part1.zip 
unzip simmc2_scene_jsons_dstc10_public.zip 
make sure the dir name is data/simmc2_scene_images_dstc10_public_part1 data/simmc2_scene_images_dstc10_public_part2 data/public  

2. data and model copy:
cd task4
download https://drive.google.com/drive/folders/1ILTFnaRTTcGWAzYXJnt_3QmeiVgE501T?usp=sharing as task4.zip
unzip task4.zip to task4 
cp task4/generate/data/oscargen_noobj_newdata/data generate/oscargen_noobj_newdata/ -fr
cp task4/generate/model/oscargen_noobj_newdata/*  generate/oscargen_noobj_newdata/ -fr
cp task4/generate/model/generagefeature/*  generate/generagefeature/ -fr
 

3. train feature:
cd task4/generate/generagefeature
chmod +x run.sh
python3 track3_gen_train.py 0 9000  (need GPU)
cp gentrain.bin ../oscargen_noobj_newdata/data/train.bin

4. train cmd:
cd task4/generate/oscargen_noobj_newdata  
python3 lasttrin.py 17e-5  240  (need GPU)


5. eval feature:
cd task4/generate/generagefeature 
python3 track3_gen_pred.py data/simmc2_dials_dstc10_devtest.json (need GPU)
cp gendev.bin  ../oscargen_noobj_newdata/data/dev.bin


6. eval cmdline:
cd task4/generate/oscargen_noobj_newdata
python3 pred_strdir.py  modelgood  find  (need GPU)
cd task4/generate
python3  format_task4_generation.py  --generation-pred-txt  oscargen_noobj_newdata/modelgood/find_0822lrnew_0.00017batch18weight3gpu1_epoch_242/checkpoint-241-42592/resulefile.txt --split-path  generagefeature/data/simmc2_dials_dstc10_devtest.json     --save-path result/subtask-4-generation.json



