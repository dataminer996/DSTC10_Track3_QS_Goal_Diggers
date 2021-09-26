import json
import jsonpath
import codecs
import sys
import pandas as pd
# did = int(did) * 1000 + index
def sys_trans_fill(dir_txt):
    txt = codecs.open(dir_txt, mode='r', encoding='utf-8')
    did = []
    caption = []
    for line in txt:
        split1 = line.split( )
        did_num = int(split1[0])
        did.append(did_num)
        line = line.replace(str(did_num), '')
        line = line.strip()
        line = line.replace(', "img_key":', '')
        line = eval(line)
        caption.append((line[0])['caption'])
    with open(sys.argv[2],'w') as tf:
       for i in range(len(did)):
          tf.writelines(str(did[i])  + "\t" + caption[i] + '\n')
         
sys_trans_fill(sys.argv[1])
