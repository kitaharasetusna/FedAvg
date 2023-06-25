import torch
import copy

def euclid(v1, v2):
    diff = v1 - v2
    return torch.matmul(diff, diff.T)
    
    
def pairwise_distance(w_locals, device):
    vectors = multi_vectorization(w_locals, device)
    distance = torch.zeros([len(vectors), len(vectors)]).to(device)
        
    for i, v_i in enumerate(vectors):
        for j, v_j in enumerate(vectors[i:]):
            distance[i][j + i] = distance[j + i][i] = euclid(v_i, v_j)
                    
    return distance    
    
def multi_vectorization(w_locals, device):
    vectors = copy.deepcopy(w_locals)
        
    for i, v in enumerate(vectors):
        for name in v:
            v[name] = v[name].reshape([-1]).to(device)
        vectors[i] = torch.cat(list(v.values()))

    return vectors