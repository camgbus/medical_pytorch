import os
import json

if __name__ == '__main__':
    labels_file_path = '/gris/gris-f/homestud/aranem/medical_pytorch-storage/JIP/preprocessed_dirs/output_test/labels.json'
    test_files_path = '/gris/gris-f/homestud/aranem/medical_pytorch-storage/JIP/preprocessed_dirs/output_test'
    file_names = [x for x in os.listdir(test_files_path) if '.json' not in x and '._' not in x]
    artefacts = ['ghosting', 'motion', 'noise', 'resolution', 'spike']
    
    with open(labels_file_path, 'r') as json_file:
        labels = json.load(json_file)

    ID_n = dict()
    OOD_n = dict()
    ID_res = {'ghosting': 0, 'motion': 0, 'noise': 0, 'resolution': 0, 'spike': 0}
    OOD_res = {'ghosting': 0, 'motion': 0, 'noise': 0, 'resolution': 0, 'spike': 0}
    ID_l = list()
    OOD_l = list()
    ID_art_count = 0
    OOD_art_count = 0
    for file in file_names:
        #if 'Decathlon' in file or 'Radiopedia' in file:
        #    continue
        ID = not('Mosmed' in file)# or 'Radiopedia' in file)
        file_perfect = True
        counted = False

        for idx, artefact in enumerate(artefacts):
            if float(labels[file+'_'+artefact]) != float(1):
                if ID:
                    ID_l.append(abs(float(labels[file+'_'+artefact])*5 - 5))
                else:
                    OOD_l.append(abs(float(labels[file+'_'+artefact])*5 - 5))

                if not counted:
                    if ID:
                        ID_n[file] = [artefact]
                        ID_res[artefact] += 1
                        ID_art_count += 1
                        counted = True
                    else:
                        OOD_n[file] = [artefact]
                        OOD_res[artefact] += 1
                        OOD_art_count += 1
                        counted = True
                else:
                    if ID:
                        ID_n[file].append(artefact)
                        ID_res[artefact] += 1
                    else:
                        OOD_n[file].append(artefact)
                        OOD_res[artefact] += 1

    ID_n['Summary'] = ID_res
    OOD_n['Summary'] = OOD_res

    results = {'ID scans (name) with artifacts': ID_n,
               'OOD scans (name) with artifacts': OOD_n,
               'ID scans artifact intensities': ID_l,
               'OOD scans artifact intensities': OOD_l,
               'Nr. of ID scans with artifacts': ID_art_count,
               'Nr. of OOD scans with artifacts': OOD_art_count,
               'ID mean artifact intensity': sum(ID_l)/len(ID_l),
               'OOD mean artifact intensity': sum(OOD_l)/len(OOD_l)}

    with open(os.path.join('/gris/gris-f/homestud/aranem/medical_pytorch-storage/data/', 'artifact_analysis.json'), 'w') as fp:
        json.dump(results, fp, sort_keys=False, indent=4)