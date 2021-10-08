import json
import glob
import argparse


def parse_prediction(file_name):
    step1_pred_data = open(file_name, 'r').readlines()

    output_result = {}
    for index, line in enumerate(step1_pred_data):
        if line:
            line_list = line.split('\t')
            pred_json = json.loads(line_list[1])
            pred_eid = str(line_list[0])
            pred_response = pred_json[0]["caption"]
            output_result[pred_eid] = pred_response

    return output_result


def embedding_prediction(file_dir):
    file_list = glob.glob(file_dir)
    merge_dict = {}
    for file_name in file_list:
        parse_result = parse_prediction(file_name)
        for eid, response in parse_result.items():
            if eid not in merge_dict.keys():
                #merge_dict[eid] = [response]
                merge_dict[eid] = response
            else:
                #merge_dict[eid].append(response)
                merge_dict[eid] =response

    return merge_dict


def main(args):
    dialogs = json.load(open(args['split_path'], 'r'))
    output_path = open(args['save_path'], 'w+')

    dialogs_output = []

    output_result = embedding_prediction(args['generation_pred_txt'])
    # print(output_result)
    for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
        dialog_idx = dialog_datum["dialogue_idx"]
        predictions = []
        #for turn_id, turn_datum in enumerate(dialog_datum["dialogue"]):
        turn_id = len(dialog_datum["dialogue"]) - 1
        turn_datum = dialog_datum["dialogue"][turn_id]
        eid = str(dialog_id * 100 + turn_id)
        predictions.append({"turn_id": turn_datum['turn_idx'], "response": output_result[eid]})
        dialogs_output.append({"dialog_id": dialog_idx, "predictions": predictions})
    json.dump(dialogs_output, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-pred-txt", help="Path dir to prediction of step1"
    )
    parser.add_argument(
        "--split-path", help="Process SIMMC file of test phase"
    )
    parser.add_argument(
        "--save-path",
        required=True, help="Path to save SIMMC dataset"
    )

    try:
        parsed_args = vars(parser.parse_args())
        main(parsed_args)
    except (IOError) as msg:
        parser.error(str(msg))
