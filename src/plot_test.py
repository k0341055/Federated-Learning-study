import os,json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

# matplotlib.use('Agg')
# color = list(mcolors.CSS4_COLORS.keys()) # mcolors.TABLEAU_COLORS.keys()
# color = list(map(lambda st: str.replace(st, "tab:", ""), color))
# plt.figure()
# plt.title('Testing Accuracy vs Communication rounds')

# path_to_json_files = '../save/json/'
# json_file_names = [filename for filename in os.listdir(path_to_json_files) if filename.endswith('.json')]
# for i,json_file_name in tqdm(enumerate(json_file_names)):
#     with open(os.path.join(path_to_json_files, json_file_name)) as json_file:
#         json_text = json.load(json_file)
#         acc = json_text['test_accuracy']
#         fn_idx = json_file_name.find('iid')
#         label_n = json_file_name[fn_idx:-5]+"({})".format(len(acc))
#         plt.plot(range(len(acc)), acc,\
#             c = color[i],ls=':',label = label_n)
# plt.legend(loc=0)
# plt.axhline(y=0.99, c='gray',ls='--')
# plt.ylabel('Testing Accuracy')
# plt.xlabel('Communication Rounds')
# plt.savefig('../save/img/comparison/test_acc_comparison.png')
# # plt.savefig('../save/img/comparison/test_acc_%s.png'%json_file_name[-15:-5])

path_to_json_files = '../save/json/'
for folder_name in tqdm(os.listdir(path_to_json_files)):
    if folder_name.startswith('.')  or 'others' in folder_name or 'desktop.ini' in folder_name: # for mac
        continue
    print('folder_name:',folder_name)
    matplotlib.use('Agg')
    color = list(mcolors.TABLEAU_COLORS.keys()) # mcolors.BASE_COLORS.keys()
    color = list(map(lambda st: str.replace(st, "tab:", ""), color))
    plt.figure()
    plt.title('Testing Accuracy vs Communication rounds')
    json_file_names = [filename for filename in os.listdir(os.path.join(path_to_json_files, folder_name)) if filename.endswith('.json')]
    for i,json_file_name in enumerate(json_file_names):
        with open(os.path.join(path_to_json_files,folder_name,json_file_name)) as json_file:
            json_text = json.load(json_file)
            acc = json_text['test_accuracy']
            fn_idx = json_file_name.find('iid')
            if 'Clusters' in json_file_name:
                c_idx = json_file_name.find('Clusters')
                label_n = "multi-center({})".format(json_file_name[c_idx+9:fn_idx-2])+json_file_name[fn_idx:-5]+"({})".format(len(acc))
            else:
                label_n = json_file_name[fn_idx:-5]+"({})".format(len(acc))
            plt.plot(range(len(acc)), acc,\
                c = color[i],ls=':',label = label_n)
            eb_idx = json_file_name.find('E[')    
            img_n = json_file_name[eb_idx:-5]
    # if not os.path.exists('../save/img/comparison/{}/'.format(img_n)):
    #     os.makedirs('../save/img/comparison/{}/'.format(img_n))
    plt.legend(loc=0)
    plt.axhline(y=0.99, c='gray',ls='--')
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Communication Rounds')
    if 'iid' in folder_name:
        plt.savefig('../save/img/comparison/test_acc_comparison_{}.png'.format(folder_name))
    elif 'multi' in folder_name:
        plt.savefig('../save/img/comparison/multi_test_acc_comparison_{}.png'.format(folder_name))
    else:
        plt.savefig('../save/img/comparison/test_acc_comparison_{}.png'.format(img_n))