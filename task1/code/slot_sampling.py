input_file = './data/finetuning_data/chunk/train.txt'
output_file = open('./data/finetuning_data/chunk/train_sample.txt', 'w+')
count = 0
with open(input_file, 'r') as f:
  lines = f.readlines()
  for line in lines:
    line = line.strip().split('\t')
    chars, pos_tags, disambiguate_label, action_label, slot_key, \
                      slot_value, objects, response = line
    if slot_key in ['type', '']:
      continue
    else:
      output_file.write('\t'.join(line) + '\n')
      count += 1
output_file.close()
print(count)