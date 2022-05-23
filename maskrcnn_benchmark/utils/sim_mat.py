import numpy as np 
import json
import sys
import scipy.misc
from PIL import Image
import json
fg_matrix = np.load('fg_matrix.npy')
fg_matrix[:,:,0] = 0
fg_matrix[0,:,:] = 0
fg_matrix[:,0,:] = 0
fg_matrix[0,0,0] = 1.0
print('fg_matrix.shape: ', fg_matrix.shape)
vg_dict = json.load(open('/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts-with-attri.json','r'))
id2pred = vg_dict['idx_to_predicate']
pred2id = vg_dict['predicate_to_idx']
label2id = vg_dict['label_to_idx']
id2label = vg_dict['idx_to_label']
pred_count = vg_dict['predicate_count']
print(pred_count)
pred_sort = sorted(pred_count.items(), key=lambda x:x[1], reverse=True)
print(pred_sort)
pred_topk_str = []
pred_topk_id = []
pred_k = 0
pred_num_pair = {}
for pred_i in pred_sort:
    if pred_k >= 15:
        break
    pred_topk_str.append(str(pred_i[0]))
    pred_topk_id.append(pred2id[str(pred_i[0])])
    pred_k = pred_k + 1

def get_sub_obj(pred, top_k = -1, num_thr = -1):
    pred_so_count = fg_matrix[:,:,pred]
    sub_num = fg_matrix.shape[0]
    obj_num = fg_matrix.shape[1]
    
    def sort_sub_obj(pred_so_count, thr_num=0):
        pred_true_sub, pred_true_obj = np.where(pred_so_count>thr_num)
        pred_ture_pairs = np.concatenate([pred_true_sub[:,None], pred_true_obj[:,None]], -1)
        pred_ind_sorts = (0 - pred_so_count).argsort(axis=None)
        pred_ind_sorts_sub = (pred_ind_sorts / sub_num).astype('int')
        pred_ind_sorts_obj = (pred_ind_sorts % sub_num).astype('int')
        pred_ind_sorts_so = np.concatenate([pred_ind_sorts_sub[:,None], pred_ind_sorts_obj[:,None]], -1)
        pred_ind_sorts_so_t = []
        for pred_so_i in pred_ind_sorts_so:
            pred_diff = pred_so_i[None,:] - pred_ture_pairs
            pred_diff = np.abs(pred_diff)
            pred_diff_sum = pred_diff.sum(-1)
            if 0 in pred_diff_sum:
                pred_ind_sorts_so_t.append(pred_so_i)
        return np.array(pred_ind_sorts_so_t)
        


        
    if top_k != -1: 
        pred_ind_sorts_so = sort_sub_obj(pred_so_count)
        if top_k > len(pred_ind_sorts_so):
            pred_num_pair[pred] = len(pred_ind_sorts_so)
            pad_so = np.ones([top_k - len(pred_ind_sorts_so),2])
            
            pad_so = pad_so * (0.0 - 1.0)
            print('hahah!')
            pred_ind_sorts_so_t = np.concatenate([pred_ind_sorts_so, pad_so], 0)
            return pred_ind_sorts_so_t
        else:
            pred_num_pair[pred] = top_k
            return pred_ind_sorts_so[:top_k]
            
    if num_thr != -1:
        pred_ind_sorts_so = sort_sub_obj(pred_so_count, num_thr=num_thr)
        
        return pred_ind_sorts_so
        
    if num_thr == -1 and top_k == -1:
        return pred_ind_sorts_so
        
def rank_dist(i,j,top_k): 
    return ((top_k-i)/top_k)*((top_k-j)/top_k)*(abs(i-j)/top_k)
    #return ((top_k-(i-1))/top_k)*(abs(i-j))
    
def find_ind(pair, pair_array):
    pair_diff = pair[None, :] - pair_array
    pair_diff = np.abs(pair_diff)
    pair_diff = pair_diff.sum(-1)
    ind_in_array = np.where(pair_diff==0)[0]
    if len(ind_in_array) > 1:
        if pair[0] != -1:
            print('bug!!!!!!, multi subject object pair:', pair,' in: ', '', pair_array)
            sys.exit()
    if len(ind_in_array) == 0:
        return None
    
    return ind_in_array[0]
    
def padarray(array,k=100):
    array_new = np.zeros([array.shape[0]*k, array.shape[1]*k])
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array_new[i*k:(i+1)*k, j*k:(j+1)*k] = array[i,j]
    return array_new
    
def sub_obj_overlap():
    pred_dist = {}
    pred_dist_array = np.zeros([len(pred2id)+1,len(pred2id)+1])
    top_k = 5
    pred_so_list = {}
    for i in range(len(pred2id)):
        it = i + 1
        pred_so_list[id2pred[str(it)]] = get_sub_obj(it, top_k=top_k, num_thr=-1)
    dis_list = []
    for i in range(len(pred2id)):
        for j in range(len(pred2id)):
            if i != j:
                it = i + 1
                jt = j + 1
                pred_so_i = pred_so_list[id2pred[str(it)]]
                pred_so_j = pred_so_list[id2pred[str(jt)]]
                dis_ij = 0
                for ii in range(len(pred_so_i)):
                    
                    pred_so_ii = pred_so_i[ii]
                    
                    if pred_so_ii[0] == -1:
                        iindj = None
                    else:
                        iindj = find_ind(pred_so_ii, pred_so_j)
                    if iindj != None:
                        dis_sub_i = rank_dist(ii*1.0, iindj*1.0, top_k=top_k)
                        dis_list.append(dis_sub_i)
                    else:
                        dis_sub_i = 1.0
                    dis_ij = dis_ij + dis_sub_i
                    
                pred_dist_array[it,jt] = dis_ij /(len(pred_so_i)*1.0)
                

    np.save('pred_sooverlap_'+str(top_k)+'.npy', pred_dist_array)
    return 

def sub_obj_overlap_set():
    pred_dist = {}
    pred_dist_array = np.zeros([len(pred2id)+1,len(pred2id)+1])
    top_k = 5
    pred_so_list = {}
    dis_list = []
    for i in range(fg_matrix.shape[-1]):
        for j in range(fg_matrix.shape[-1]):
            if i != j:
                obj_i, sub_i = np.where(fg_matrix[:,:,i] > 0)
                pair_i = np.concatenate([obj_i[:,None], sub_i[:,None]], -1)
                len_i = len(pair_i)
                obj_j, sub_j = np.where(fg_matrix[:,:,j] > 0)
                pair_j = np.concatenate([obj_j[:,None], sub_j[:,None]], -1)
                len_j = len(pair_j)
                if len_j > len_i:
                    pair_s = pair_i
                    pair_l = pair_j
                    ind_s = i
                    ind_l = j
                else:
                    pair_s = pair_j
                    pair_l = pair_i
                    ind_s = j
                    ind_l = i
                over_lap_count = 0
                pred_ind_sorts_so_t = []
                for pair_sk in pair_s:
                    pred_diff = pair_sk[None,:] - pair_l
                    pred_diff = np.abs(pred_diff)
                    pred_diff_sum = pred_diff.sum(-1)
                    if 0 in pred_diff_sum:
                        pred_ind_sorts_so_t.append(pair_sk)
                if len(pair_l) > 0:
                    pred_dist_array[ind_l, ind_s] = len(pred_ind_sorts_so_t) /(len(pair_l)*1.0)
                if len(pair_s) > 0:
                    pred_dist_array[ind_s, ind_l] = len(pred_ind_sorts_so_t) /(len(pair_s)*1.0)
            if i == j:
                pred_dist_array[i,j]=1.0
    #np.save('pred_sooverlapset.npy', pred_dist_array)
    return pred_dist_array
    
def been_coverage(pred_i):
    pred_so_i = fg_matrix[:,:,pred_i]
    pred_so_i_inds = np.where(pred_so_i>0)
    cover_i = np.zeros([fg_matrix.shape[-1]])
    for j in range(fg_matrix.shape[-1]):
        cover_j  = 0
        if j != pred_i:
            pred_so_freq_j = fg_matrix[:,:,j]
            for k in range(len(pred_so_i_inds[0])):
                if pred_so_freq_j[pred_so_i_inds[0][k],pred_so_i_inds[1][k]] >= pred_so_i[pred_so_i_inds[0][k],pred_so_i_inds[1][k]]:
                    cover_j = cover_j + 1
        cover_i[j] = cover_j / (len(pred_so_i_inds[0]) + 1e-8)
    return cover_i
    
def countij_cover():
    cover_ij = np.zeros([fg_matrix.shape[-1],fg_matrix.shape[-1]])
    for i in range(cover_ij.shape[0]):
        cover_ij[i,:] = been_coverage(i)
    cover_i = cover_ij.max(-1)
    cover_j_ind = cover_ij.argmax(-1)
    cover_i_sort = (cover_i).argsort()
    print(cover_i)
    cover_sort_list = []
    for i in cover_i_sort:
        if i != 0:
            print(id2pred[str(i)],'->',id2pred[str(cover_j_ind[i])], end=' ,')
            cover_sort_list.append(id2pred[str(i)])
            
    print('')    
    json.dump(cover_sort_list, open('pred_cover_sort.json','w'))
            
    
def w2v_dist():
    pred_w2v = np.load('predicates_w2v.npy')
    pred_dist_array = np.zeros([len(pred2id)+1,len(pred2id)+1])
    for i in range(pred_w2v.shape[0]):
        for j in range(pred_w2v.shape[0]):
            if i == 0 or j == 0:
                continue
            pred_dist_array[i,j] = ((pred_w2v[i]-pred_w2v[j])*(pred_w2v[i]-pred_w2v[j])).mean()
            #pred_dist_array[i,j] = np.dot(pred_w2v[i], pred_w2v[j])/(np.linalg.norm(pred_w2v[i])* np.linalg.norm(pred_w2v[j]))
    pred_dist_array = (pred_dist_array - pred_dist_array.min(-1)[:,None]) / (pred_dist_array.max(-1) - pred_dist_array.min(-1)+1e-6)[:,None]
    np.save('pred_dist_array_w2v.npy', pred_dist_array)
    image_array = padarray(pred_dist_array, 100) * 255 
    im = Image.fromarray(image_array)
    im = im.convert('L') 
    im.save('pred_dist_array_w2v.png')

    flag = 0
    for i in range(len(pred_dist_array)):
        for j in range(len(pred_dist_array)):
            if abs(pred_dist_array[i,j] - pred_dist_array[j,i]) > 1e-5:
                flag = 1
                print(pred_dist_array[i,j])
                print(pred_dist_array[j,i])
    print('symm flag: ', flag)
    pred_near = {}
    for i in range(len(pred_dist_array)):
        if i == 0:
            continue
        pred_dist_i = pred_dist_array[i]
        pred_dist_i_sort_ind = (pred_dist_i).argsort()
        print('pred: ', id2pred[str(i)])
        pred_near[id2pred[str(i)]] = []
        print('near pred sort: ', )
        for j in pred_dist_i_sort_ind[:6]:
            if str(j) in id2pred:
                print(id2pred[str(j)],', ', end='')
                pred_near[id2pred[str(i)]].append(id2pred[str(j)])
        print('\n')
    json.dump(pred_near, open('pred_near_w2v.json','w'))

def prob_sim():
    pred_dist_array = np.zeros([len(pred2id)+1,len(pred2id)+1])
    for i in range(fg_matrix.shape[-1]):
        for j in range(fg_matrix.shape[-1]):
            prob_i = fg_matrix[:,:,i]
            prob_i = prob_i.reshape(-1)
            prob_i = prob_i / (prob_i.sum() + 1e-8)
            prob_j = fg_matrix[:,:,j]
            prob_j = prob_j.reshape(-1)
            prob_j = prob_j / (prob_j.sum() + 1e-8)
            pred_dist_array[i,j] = np.linalg.norm(prob_i - prob_j, ord=2)
    pred_dist_array = pred_dist_array.max() - pred_dist_array
    np.save('pred_soprob.npy', pred_dist_array)

def so_entropy_pred():
    fg_mat = fg_matrix
    fg_mat_r = fg_mat.reshape(-1, 51)
    pred_so_freq = fg_mat_r.transpose([1, 0])
    fg_mat_mask = (pred_so_freq > 0.).astype('float')
    pred_so_dist = pred_so_freq / (pred_so_freq.sum(-1)[:, None] + 1e-8)
    pred_so_ent = 0.0 - ((pred_so_dist * np.log(pred_so_dist + 1e-8))).sum(-1)
    pred_so_ent_sort = (0 - pred_so_ent).argsort()
    pred_topk = []
    pred_topk_ent = []
    pred_count = 0
    id2pred['0'] = "_bg_"
    for pred_i in pred_so_ent_sort:
        if pred_count >= 15:
            break
        
        pred_topk.append(str(id2pred[str(pred_i)]))
        pred_topk_ent.append(pred_so_ent[pred_i])
        pred_count = pred_count + 1
    print(pred_topk)
    print(pred_topk_ent)
if __name__ == "__main__":
    
    #w2v_dist()
    #sub_obj_overlap()
    #sub_obj_overlap_set()
    prob_sim()
    # countij_cover()
    #so_entropy_pred()
    
