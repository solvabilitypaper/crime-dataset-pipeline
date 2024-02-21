"""*
 *     Crime Dataset Pipeline
 *
 *        File: step4_result_parser.py
 *
 *     Authors: Deleted for purposes of anonymity
 *
 *     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
 *
 * The software and its source code contain valuable trade secrets and shall be maintained in
 * confidence and treated as confidential information. The software may only be used for
 * evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
 * license agreement or nondisclosure agreement with the proprietor of the software.
 * Any unauthorized publication, transfer to third parties, or duplication of the object or
 * source code---either totally or in part---is strictly prohibited.
 *
 *     Copyright (c) 2021 Proprietor: Deleted for purposes of anonymity
 *     All Rights Reserved.
 *
 * THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT
 * DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION.
 *
 * NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
 * IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE
 * LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
 * FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
 * OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
 * ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
 * TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGES.
 *
 * For purposes of anonymity, the identity of the proprietor is not given herewith.
 * The identity of the proprietor will be given once the review of the
 * conference submission is completed.
 *
 * THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 *"""

import time
from os.path import isfile, join
from os import listdir
import statistics
import json
import copy

import numpy as np


def main():
    results_dir = '../../data/crime-knowledge-graph/'
    result_file = '.results_feature'

    sheet = []
    header = np.array(['Crime_Type', 'District', 'Type', 'Source',
                       'Pos_Train_Samples', 'Neg_Train_Samples',
                       'Pos_Valid_Samples', 'Neg_Valid_Samples',
                       'Pos_Test_Samples', 'Neg_Test_Samples', 'Classifier',
                       'F1_S_mean', 'F1_S_std', 'F1_S_median', 'F1_U_mean',
                       'F1_U_std', 'F1_U_median', 'BA_mean', 'BA_std',
                       'BA_median', 'AUROC_mean', 'AUROC_std', 'AUROC_median',
                       'PS_macro_mean', 'PS_macro_std', 'PS_macro_median',
                       'PS_micro_mean', 'PS_micro_std', 'PS_micro_median',
                       'Pre_T070_mean', 'Pre_T070_std', 'Pre_T070_median',
                       'Rec_T070_mean', 'Rec_T070_std', 'Rec_T070_median',
                       'Pre_T080_mean', 'Pre_T080_std', 'Pre_T080_median',
                       'Rec_T080_mean', 'Rec_T080_std', 'Rec_T080_median',
                       'Pre_T090_mean', 'Pre_T090_std', 'Pre_T090_median',
                       'Rec_T090_mean', 'Rec_T090_std', 'Rec_T090_median',
                       'Pre_T095_mean', 'Pre_T095_std', 'Pre_T095_median',
                       'Rec_T095_mean', 'Rec_T095_std', 'Rec_T095_median'])
    sheet.append(header)

    # load existing files
    only_files = [file for file in listdir(results_dir) if isfile(join(results_dir, file))]
    identifiers = {file for file in only_files if result_file in file}
    identifiers = list(identifiers)
    identifiers.sort()

    # summary by crime_type
    summary = {}

    # parse files
    for identifier in identifiers:
        print(identifier)
        crime_type = identifier[identifier.find('_') + 1:]
        crime_type = crime_type[:crime_type.find('_')]
        district = identifier[identifier.find('_') + 1:identifier.find('.')]
        sum_source = identifier.replace(district[district.find('_') + 1:], 'XXXX')
        district = district[district.find('_') + 1:].replace('_', ' ')
        data_type = 'graph' if 'graph' in identifier.lower() else 'tabular'
        line = [crime_type, district, data_type, identifier]

        sum_key = (crime_type, 'all', data_type, sum_source)
        if sum_key not in summary:
            summary[sum_key] = {'Num Positive Train Samples': 0,
                                'Num Negative Train Samples': 0,
                                'Num Positive Valid Samples': 0,
                                'Num Negative Valid Samples': 0,
                                'Num Positive Test Samples': 0,
                                'Num Negative Test Samples': 0,
                                'test': {}}
            del summary[sum_key]['test']  # workaround to silence a warning

        # load file
        with open(join(results_dir, identifier), 'r', encoding='utf-8') as obj:
            results = json.loads(obj.read())

        if not bool(results):  # if the dict is empty, we only had negative samples
            line += 49 * ['N/A']
            sheet.append(np.array(line))
            continue

        # record number of training, validation and test samples
        line.append(results['Num Positive Train Samples'])
        summary[sum_key]['Num Positive Train Samples'] += int(results['Num Positive Train Samples'])
        if ', ' in results['Num Train Samples']:
            tmp = int(results['Num Train Samples'][1:-1].replace(',', '').split()[0])
        else:
            tmp = int(results['Num Train Samples'])
        tmp -= int(results['Num Positive Train Samples'])
        line.append(tmp)
        summary[sum_key]['Num Negative Train Samples'] += tmp
        line.append(results['Num Positive Valid Samples'])
        summary[sum_key]['Num Positive Valid Samples'] += int(results['Num Positive Valid Samples'])
        if ', ' in results['Num Valid Samples']:
            tmp = int(results['Num Valid Samples'][1:-1].replace(',', '').split()[0])
        else:
            tmp = int(results['Num Valid Samples'])
        tmp -= int(results['Num Positive Valid Samples'])
        line.append(tmp)
        summary[sum_key]['Num Negative Valid Samples'] += tmp
        line.append(results['Num Positive Test Samples'])
        summary[sum_key]['Num Positive Test Samples'] += int(results['Num Positive Test Samples'])
        if ', ' in results['Num Test Samples']:
            tmp = int(results['Num Test Samples'][1:-1].replace(',', '').split()[0])
        else:
            tmp = int(results['Num Test Samples'])
        tmp -= int(results['Num Positive Test Samples'])
        line.append(tmp)
        summary[sum_key]['Num Negative Test Samples'] += tmp

        # parse file
        classifiers = {key for key, value in results.items() if isinstance(value, dict)}
        classifiers = list(classifiers)
        classifiers.sort()
        for classifier in classifiers:
            sub_line = [classifier]

            tmp_values = {'f1_s_values': [],
                          'f1_u_values': [],
                          'ba_values': [],
                          'auroc_values': [],
                          'ps_macro': [],
                          'ps_micro': [],
                          'pre_t070_values': [],
                          'rec_t070_values': [],
                          'pre_t080_values': [],
                          'rec_t080_values': [],
                          'pre_t090_values': [],
                          'rec_t090_values': [],
                          'pre_t095_values': [],
                          'rec_t095_values': []}

            if classifier not in summary[sum_key]:
                summary[sum_key][classifier] = copy.deepcopy(tmp_values)

            for run_id, values in results[classifier].items():
                if not bool(values):  # check if the dict is empty
                    continue

                # the threshold values were determined based on the validation data
                best_threshold = f'Threshold_{values["Threshold"]}'

                tmp_values['f1_s_values'].append(values[best_threshold]['F1 (Class 1)'])
                tmp_values['f1_u_values'].append(values[best_threshold]['F1 (Class 0)'])
                tmp_values['ba_values'].append(values[best_threshold]['B. Accuracy'])
                tmp_values['auroc_values'].append(values['ROCAUC_Score'])
                tmp_values['ps_macro'].append(values[best_threshold]['PS_Macro'])
                tmp_values['ps_micro'].append(values[best_threshold]['PS_Micro'])
                tmp_values['pre_t070_values'].append(values['Threshold_0.3']['Precision (Class 0)'])
                tmp_values['rec_t070_values'].append(values['Threshold_0.3']['Recall (Class 0)'])
                tmp_values['pre_t080_values'].append(values['Threshold_0.2']['Precision (Class 0)'])
                tmp_values['rec_t080_values'].append(values['Threshold_0.2']['Recall (Class 0)'])
                tmp_values['pre_t090_values'].append(values['Threshold_0.1']['Precision (Class 0)'])
                tmp_values['rec_t090_values'].append(values['Threshold_0.1']['Recall (Class 0)'])
                tmp_values['pre_t095_values'].append(values['Threshold_0.05']['Precision (Class 0)'])
                tmp_values['rec_t095_values'].append(values['Threshold_0.05']['Recall (Class 0)'])

                summary[sum_key][classifier]['f1_s_values'].append(values[best_threshold]['F1 (Class 1)'])  # TODO
                summary[sum_key][classifier]['f1_u_values'].append(values[best_threshold]['F1 (Class 0)'])
                summary[sum_key][classifier]['ba_values'].append(values[best_threshold]['B. Accuracy'])
                summary[sum_key][classifier]['auroc_values'].append(values['ROCAUC_Score'])
                summary[sum_key][classifier]['ps_macro'].append(values[best_threshold]['PS_Macro'])
                summary[sum_key][classifier]['ps_micro'].append(values[best_threshold]['PS_Micro'])
                summary[sum_key][classifier]['pre_t070_values'].append(values['Threshold_0.3']['Precision (Class 0)'])
                summary[sum_key][classifier]['rec_t070_values'].append(values['Threshold_0.3']['Recall (Class 0)'])
                summary[sum_key][classifier]['pre_t080_values'].append(values['Threshold_0.2']['Precision (Class 0)'])
                summary[sum_key][classifier]['rec_t080_values'].append(values['Threshold_0.2']['Recall (Class 0)'])
                summary[sum_key][classifier]['pre_t090_values'].append(values['Threshold_0.1']['Precision (Class 0)'])
                summary[sum_key][classifier]['rec_t090_values'].append(values['Threshold_0.1']['Recall (Class 0)'])
                summary[sum_key][classifier]['pre_t095_values'].append(values['Threshold_0.05']['Precision (Class 0)'])
                summary[sum_key][classifier]['rec_t095_values'].append(values['Threshold_0.05']['Recall (Class 0)'])

            for _, value in tmp_values.items():  # our dict has a fixed order
                if len(value) == 0:
                    sub_line.append('N/A')
                    sub_line.append('N/A')
                    sub_line.append('N/A')
                elif len(value) > 1:
                    sub_line.append(statistics.mean(value))
                    sub_line.append(statistics.stdev(value))
                    sub_line.append(statistics.median(value))
                else:
                    sub_line.append(value[0])
                    sub_line.append('N/A')
                    sub_line.append(value[0])

            sheet.append(np.array(line + sub_line))

    # save summary
    for sum_key, values in summary.items():
        line = list(sum_key)
        line.append(values['Num Positive Train Samples'])
        line.append(values['Num Negative Train Samples'])
        line.append(values['Num Positive Valid Samples'])
        line.append(values['Num Negative Valid Samples'])
        line.append(values['Num Positive Test Samples'])
        line.append(values['Num Negative Test Samples'])

        for classifier, values2 in values.items():
            if not isinstance(values2, dict):
                continue
            sub_line = [classifier]

            for _, values3 in values2.items():  # our dict has a fixed order
                if len(values3) == 0:
                    sub_line.append('N/A')
                    sub_line.append('N/A')
                    sub_line.append('N/A')
                elif len(values3) > 1:
                    sub_line.append(statistics.mean(values3))
                    sub_line.append(statistics.stdev(values3))
                    sub_line.append(statistics.median(values3))
                else:
                    sub_line.append(values3[0])
                    sub_line.append('N/A')
                    sub_line.append(values3[0])

            sheet.append(np.array(line + sub_line))

    sheet = np.array(sheet)
    np.savetxt(join(results_dir, 'results.csv'), sheet, delimiter=',', fmt="%s")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f'Runtime: {time.time() - start}')
