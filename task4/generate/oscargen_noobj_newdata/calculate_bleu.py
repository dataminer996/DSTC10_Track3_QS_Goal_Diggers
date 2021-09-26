import sys
import json
import nltk
import numpy as np


def normalize_sentence(sentence):
    """Normalize the sentences and tokenize."""
    return nltk.tokenize.word_tokenize(sentence.lower())


def proc_result(file_path):
    result = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 0:
                line_list = line.split('\t')
                eid = line_list[0]
                for i in range(1, len(line_list)):
                    if line_list[i]:
                        # print('eid', line_list[i])
                        response = json.loads(line_list[i])
                        # print(response[0])
                        # print(len(response))
                        result[int(eid)] = response[0]['caption']
    return result


def read_dialogue(file_path):
    result = {}
    with open(file_path, "r") as simmc_file:
        dialogs = json.load(simmc_file)
    count = 0
    for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
        for turn_id, turn_datum in enumerate(dialog_datum["dialogue"]):
            eid = count*100 + turn_id
            response = turn_datum["system_transcript"]
            result[eid] = response
        count += 1
    return result


def calculate_bleu(target_path, pred_path):
    # Compute BLEU scores.
    bleu_scores = []
    # Smoothing function.
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    pred_dict = proc_result(pred_path)
    true_dict = read_dialogue(target_path)
    for eid, response in true_dict.items():
        try:
            bleu_score = nltk.translate.bleu_score.sentence_bleu(
                [normalize_sentence(true_dict[eid])],
                normalize_sentence(pred_dict[eid]),
                smoothing_function=chencherry.method7,
            )
        except:
            
            #if pred_dict[eid].lower() == true_dict[eid].lower():
            #    bleu_score = 1
            #else:
                bleu_score = 0
        #print(bleu_score)
        bleu_scores.append(bleu_score)
    #print(bleu_scores)
    #print(
    #    "BLEU score: {} +- {}".format(
    #        np.mean(bleu_scores), np.std(bleu_scores) / np.sqrt(len(bleu_scores))
    #    )
    #)
    return np.mean(bleu_scores), np.std(bleu_scores) / np.sqrt(len(bleu_scores))


if __name__ == '__main__':
    #target_path = '../data/simmc2_dials_dstc10_devtest.json'
    target_path = sys.argv[1]
    #pred_path = '../data/resulefile.txt'
    pred_path = sys.argv[2]
    calculate_bleu(target_path, pred_path)
