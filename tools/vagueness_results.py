import torch
import numpy as np
import copy
def cal_vagueness(model_path, model_name):
    eval_results = torch.load(model_path+"/"+model_name+"/result_dict.pytorch")
    conf_mat = eval_results['predicate_confusion_matrix'][1:,1:]
    diag_ind = []
    for i in range(conf_mat.shape[0]):
        diag_ind.append([i,i])
    conf_mat = conf_mat / conf_mat.sum(-1)[:,None]
    diag_ind = np.array(diag_ind)
    conf_mat_diag = copy.deepcopy(np.diagonal(conf_mat))
    print(np.max(conf_mat/(conf_mat_diag+1e-2)))
    print(np.min(conf_mat/(conf_mat_diag+1e-2)))
    conf_mat[diag_ind[:,0], diag_ind[:,1]] = -1.0

    conf_mat_sort = 0.0 - np.sort((0.0 - conf_mat), -1)
    top_k = [1,3,5,10,20, 30]
    vag_k = {}
    for top_i in top_k:
        vag_all = conf_mat_diag[:,None] - conf_mat_sort[:,:top_i]
        vag_all = vag_all * (vag_all >= 0.0)
        vag_k['top_'+str(top_i)] = vag_all.mean(-1).mean(-1)
    print(model_name)
    print(vag_k)
    
    
    
    
    
    


if __name__ == "__main__":
    model_name_list = ["transformer"]
    model_path = "/mnt/hdd1/lvxinyu/datasets/visual_genome/model/checkpoints/"
    for model_name_i in model_name_list:
        cal_vagueness(model_path, model_name_i)
    