#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch_geometric.transforms as T
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import numpy as np
import random
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import os
from ogb.graphproppred import PygGraphPropPredDataset
import torch
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import os
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.utils import degree


# In[2]:


import sys
sys.set_int_max_str_digits(0)


# In[3]:




# In[4]:


class ConcatPos(object):
    def __call__(self, data):
        if data.edge_attr is not None:
            data.edge_attr = None
        data.x = torch.cat([data.x, data.pos], dim=1)
        data.pos = None
        return data


# In[5]:


import torch
import numpy as np
import os
import random
import sys


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return 

seed_everything(0)


# In[6]:


class Squeeze(object):
    def __call__(self, data):

        data.y = data.y.squeeze(0)
        # data.x = data.x.float()
        
        return data
    
    
    
import torch
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
datasetname = 'IMDB-BINARY'

# if datasetname in ['DD', 'MUTAG', 'NCI1']:
dataset = TUDataset('data/TUDataset', name=datasetname)#, transform=T.Compose([Complete()]), use_node_attr=True)
        
# if datasetname in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace','ogbg-molbbbp']:
            
#             dataset = PygGraphPropPredDataset(name=datasetname, transform = T.Compose([Squeeze()]))#, transform=T.Compose([RemoveEdgeAttr()]))

path = 'data/'+datasetname

# transform = T.Compose([ConcatPos()])
# train_dataset= GNNBenchmarkDataset(path, name=datasetname, split='train', transform=transform)
# val_dataset= GNNBenchmarkDataset(path, name=datasetname, split='val', transform=transform)
# test_dataset= GNNBenchmarkDataset(path, name=datasetname, split='test', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
# nnodes = [x.num_nodes for x in train_dataset]
# print('mean #nodes:', np.mean(nnodes), 'max #nodes:', np.max(nnodes))
            
print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[5]  # Get the fifth graph object.


# In[7]:


data


# In[8]:


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
dataset = dataset.shuffle()


# In[9]:


# split_idx = dataset.get_idx_split()


# In[76]:


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
datasettemp = []


# In[11]:


# dataset = dataset.to(device)


# In[12]:


if datasetname == 'IMDB-BINARY':#'PROTEINS' or datasetname == 'DD' or datasetname == 'NCI1': #or datasetname == 'ogbg-molbbbp':
    for index in range(0, len(dataset)):
        
        
        # data = Data(x=torch.ones(dataset[index].num_nodes,1).to(torch.long).to(device), edge_index=dataset[index].edge_index, y = dataset[index].y, edge_attr= torch.ones(dataset[index].edge_index.shape[1], 1).to(torch.long)).to(device)
        deg_x = degree(dataset[index].edge_index[0], dataset[index].num_nodes).reshape(-1,1)
        data = Data(x=deg_x.to(torch.long).to(device), edge_index=dataset[index].edge_index, y = dataset[index].y, edge_attr= torch.ones(dataset[index].edge_index.shape[1], 1).to(torch.long)).to(device)

        # data.edge_attr =
        datasettemp.append(data)
        # print((data.y))
        # datasettemp[index]
    dataset = datasettemp


# In[13]:


n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
print('n', n)
print('test_dataset', len(test_dataset))
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
print('train_dataset', len(train_dataset))


# In[14]:


edge_attrs = set()
for d in train_dataset:
    edge_attr = d.edge_attr
    [edge_attrs.add(tuple(edge_attr[ea, :].tolist())) for ea in range(edge_attr.shape[0])]
print(len(edge_attrs))
'Suspicion confirmed. Only 11 of 60 (5x6x2) possible configurations found in the dataset.'


# In[15]:


node_attrs = set()
for d in train_dataset:
    nl = d.x#[:,0]
    [node_attrs.add(tuple(nl[ea, :].tolist())) for ea in range(nl.shape[0])]
print(len(node_attrs))


# In[16]:


node_labels = {n:idx for idx, n in enumerate(node_attrs)}
node_labels_rev = {idx:n for idx, n in enumerate(node_attrs)}


# In[17]:


edge_labels = {edge_config:idx for idx,edge_config in enumerate(edge_attrs)}
edge_labels_rev = {idx:edge_config for idx,edge_config in enumerate(edge_attrs)}


# In[18]:


edge_attrs
count_unique_edge_labels = len(edge_attrs)


# In[19]:


count_unique_node_labels = len(node_labels)


# In[20]:


count_unique_node_labels


# In[21]:


count_unique_edge_labels


# In[22]:


edge_labels_rev


# In[23]:


a_1 = []
a_2 = []
for data in dataset:
    # print(data.y)
    if data.y == 1:
        a_1.append(data.y)
    else:
        a_2.append(data.y)
print(len(a_1),len(a_2))


# In[24]:


dataset[0].x.dtype


# In[25]:


# node_labels


# In[26]:


train_dataset_processed = []
for d in train_dataset:
    data = Data(
                x=F.one_hot(torch.tensor([node_labels[tuple(d.x[ea,:].tolist())] for ea in range(d.x.shape[0])], dtype=torch.long),num_classes = count_unique_node_labels),

                original_x = d.x,
                edge_index=d.edge_index,
                y=d.y,
                original_edge_attr = d.edge_attr,
               
                edge_attr=F.one_hot(torch.tensor([edge_labels[tuple(d.edge_attr[ea,:].tolist())] for ea in range(d.edge_attr.shape[0])], dtype=torch.long),num_classes = count_unique_edge_labels)
               
               )
    train_dataset_processed.append(data)


# In[27]:


data


# In[28]:


actual_orig_dataset=train_dataset


# In[ ]:





# In[29]:


def get_MPTree(Graph, root_node, hops):
    assert hops < 5, f"Number of hops {hops} is too computationally extensive for a proof of concept"
    G = Graph.to_undirected()
    MPTree = nx.DiGraph()
    def inf_counter():
        ctr = 0
        while True:
            yield ctr
            ctr += 1
    ctr = inf_counter()
    start = root_node
    hop = 0
    Q = [(start, next(ctr), hop)]
    MPTree.add_node(Q[0][1], nodeorigid=start, label = G.nodes[start]['label'])

    while hop < hops and len(Q)>0:
        # print(hop, 'Q', Q)
        top, topid, hop = Q.pop(0)
        # print('t, tid, h', top, topid, hop)
        if hop >= hops:
            continue
        neighbors = G.neighbors(top)
        # print('nbrs',list( neighbors))
        for neighbor, new_id in zip(neighbors, ctr):
            MPTree.add_node(new_id, nodeorigid=neighbor, label=G.nodes[neighbor]['label'])
            MPTree.add_edge(topid, new_id, label=G.get_edge_data(top, neighbor)['label'])
            Q.append((neighbor, new_id, hop+1))
            # print('nbr', neighbor)
            
    #sm
    
    # MPTree = nx.reverse(MPTree)
    return MPTree


# In[30]:


def get_labels(G):
    for node1, node2, data in G.edges.data():
        # print('data ', data)
        
        roundarray = np.around(data['edge_attr']) 
        label_value = np.where(roundarray==1)[0][0]
        G.edges[node1, node2]["label"] = label_value
        G.edges[node2, node1]['label'] = label_value
        
    for i in range(0, len(G)):
        roundarray = np.around(G.nodes[i]['x'])
        label_value = np.where(roundarray==1)[0][0]
        G.nodes[i]['label'] = label_value
    return G

# In[31]:


def to_nx(data):
    # print('data ', data)
    G = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'])
    return G
def get_dfscode(G):
    G_ = G.to_undirected()
    dfs_code = get_min_dfscode(G_)
    dfs_code = list(map(''.join, dfs_code))
    return ''.join(dfs_code)


# In[32]:


import pyfpgrowth


# In[33]:


def get_can_lab_tree(tree, root=0):
    # leaves = filter(lambda x:tree.out_degree(x)==0,tree)
    # rtree = tree.reverse()
    # ^^ to check
    #
    # base case
    if tree.out_degree(root) == 0:
        return int(f'1{tree.nodes[root]["label"]+2}0')
    child_labels = []
    for child in tree.adj[root]:
        child_label = get_can_lab_tree(tree, child)
        edge_label = tree.get_edge_data(root, child)['label']
        child_labels.append(int(f'{child_label}{edge_label+2}'))
    child_labels = sorted(child_labels)
    can_lab = int(f'1{tree.nodes[root]["label"]+2}{"".join(str(c) for c in child_labels)}0')

    return can_lab


# In[34]:


num_hops = 2
dfscodes = {}
ctr = 0


graph_MPTree_dict = {}
labels_graph_dict = {}
labels_trees_to_graph_dict = {}

map_label_to_mptree = {}

counter_index = 0 
for index_graph, data in enumerate(train_dataset_processed):
    counter_index+=1

    if counter_index%10==0:
        print('counter_index', counter_index)
        pass    
    
    MPTrees = []
    ctr += 1
    # preprocessing
    G = to_nx(data)
    G = get_labels(G)
        

    for node in G.nodes:
        MPTree = get_MPTree(G, node, num_hops)
        if len( MPTree) <=1:
            print('len mptree ', len(MPTree))

            continue
        MPTrees.append(MPTree)
        
        
    graph_MPTree_dict[index_graph]=MPTrees
    
    labels_trees_this_graph = list(map(get_can_lab_tree,MPTrees))
    

    for label, MPTree in zip(labels_trees_this_graph, MPTrees):
        map_label_to_mptree[label] = MPTree.reverse()
       
    
    labels_graph_dict[index_graph] = labels_trees_this_graph
    
    for label in labels_trees_this_graph:
        if label not in labels_trees_to_graph_dict:
            labels_trees_to_graph_dict[label]=[]
        
        labels_trees_to_graph_dict[label].append(index_graph)
    
print(len(MPTrees))


# In[35]:


label


# In[36]:


g=map_label_to_mptree[label]


# In[37]:


g.nodes(data=True)


# In[38]:


g.edges(data=True)


# In[ ]:





# In[ ]:





# In[39]:

# In[ ]:


import pickle

# a = {'hello': 'world'}

with open('processed/{}'.format(datasetname)+'_num_hop{}_labels_trees_to_graph_dict.pickle'.format(num_hops), 'wb') as handle:
    pickle.dump(labels_trees_to_graph_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('processed/{}'.format(datasetname)+'_num_hop{}_map_label_to_mptree.pickle'.format(num_hops), 'wb') as handle:
    pickle.dump(map_label_to_mptree, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('processed/{}'.format(datasetname)+'_num_hop{}_labels_graph_dict.pickle'.format(num_hops), 'wb') as handle:
    pickle.dump(labels_graph_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# with open('processed/{}'.format(datasetname)+'_num_hop{}_graph_MPTree_dict.pickle'.format(num_hops), 'wb') as handle:
#     pickle.dump(graph_MPTree_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    


# In[ ]:


# num_hops = 1


# In[ ]:


# with open('processed/{}'.format(datasetname)+'_num_hop{}_labels_trees_to_graph_dict.pickle'.format(num_hops), 'rb') as handle:
#     labels_trees_to_graph_dict = pickle.load(handle)

# with open('processed/{}'.format(datasetname)+'_num_hop{}_map_label_to_mptree.pickle'.format(num_hops), 'rb') as handle:
#     map_label_to_mptree = pickle.load(handle)

    
# with open('processed/{}'.format(datasetname)+'_num_hop{}_labels_graph_dict.pickle'.format(num_hops), 'rb') as handle:
#     labels_graph_dict = pickle.load(handle)


# In[ ]:





# In[40]:


graph_class_dict = {}
class_graph_dict = {}


# In[41]:


len(labels_graph_dict.keys())


# In[42]:


for graph_index, labels_of_graph in labels_graph_dict.items():
    graph_class = train_dataset_processed[graph_index].y.item()
    graph_class_dict[graph_index] = graph_class
 
    if train_dataset_processed[graph_index].y.item() not in class_graph_dict:
        
        class_graph_dict[train_dataset_processed[graph_index].y.item()]= []
    
    class_graph_dict[train_dataset_processed[graph_index].y.item()].append(graph_index)
    


# In[43]:


len(class_graph_dict[0]), len(class_graph_dict[1])


# In[44]:


class_code_labels = {}
labels_to_class ={}


# In[45]:


for graph_index, labels_of_graph in labels_graph_dict.items():
    class_graph = graph_class_dict[graph_index]
    if class_graph not in class_code_labels:
        class_code_labels[class_graph]= []
    
    class_code_labels[class_graph].extend(labels_of_graph)


# In[46]:


tree_ctr_class = {}

for classid, labels in class_code_labels.items():# zip(labels, MPTrees):
    
    if classid not in tree_ctr_class:
        tree_ctr_class[classid] = {}
        
    for lab in labels:

        if lab not in tree_ctr_class[classid]:
            tree_ctr_class[classid][lab]=1

        tree_ctr_class[classid][lab]+=1


# In[47]:


top_MPTrees_class = {}
topLabels_classwise = {}
freq_thr = {0:1, 1: 1}

for classid, lab_freq in tree_ctr_class.items():
    
    if classid not in top_MPTrees_class:
        top_MPTrees_class[classid] = []
        topLabels_classwise[classid] = []
        
    
    for lab, freq in lab_freq.items():
        
        if freq>freq_thr[classid]:
            
            tree_of_label = map_label_to_mptree[lab]
            top_MPTrees_class[classid].append(tree_of_label)
            topLabels_classwise[classid].append(lab)
            


# In[48]:


len(topLabels_classwise[0]), len(topLabels_classwise[1])


# In[49]:


transactions_classwise = {}
for classid, labels in topLabels_classwise.items():
    transactions_classwise[classid]= {}
    
    for lab in labels:
        
        graphidlist = labels_trees_to_graph_dict[lab]
        for graphid in graphidlist:
            if graphid in class_graph_dict[classid]:
                if graphid not in transactions_classwise[classid]:
                    transactions_classwise[classid][graphid] = set()
            
            
                transactions_classwise[classid][graphid].add(lab)
        


# In[50]:


len(transactions_classwise[0]), len(transactions_classwise[1])


# In[51]:


transactions_array_classwise = {}
for classid, transactions in  transactions_classwise.items():
    # list_transactions = list(transactions.values())
    print(classid)
    list_transactions = []
    for i in transactions.values():
        list_transactions.append(list(i))
        
    transactions_array_classwise[classid] = list_transactions


# In[52]:


import pyfpgrowth
from torch_geometric.utils.convert import from_networkx


# In[53]:


patterns_dict  = {}

#60
threshold_class_dict  = {0:6
                         ,1:10
                        }


# 64
# threshold_class_dict  = {0:4
#                          ,1:7
#                         }


# threshold_class_dict  = {0:45
#                          ,1:35
#                         }

#gr8
# threshold_class_dict  = {0:7
#                          ,1:10
#                         }

# #v good 63 65
threshold_class_dict  = {0:11
                         ,1:12
                        }


# threshold_class_dict  = {0:8
#                          ,1:14
#                         }

# threshold_class_dict  = {0:9
#                          ,1:20
#                         }



for classid, transactions in transactions_array_classwise.items():
    print('classid', classid)
    patterns_dict[classid] = pyfpgrowth.find_frequent_patterns(transactions, threshold_class_dict[classid])
    
    for pattern, freq in patterns_dict[classid].items():
        patterns_dict[classid][pattern] = freq#*1.0/len(transactions_array_classwise[classid])
        
        


# In[75]:


device


# In[77]:


from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


# In[78]:


from torch_geometric.loader import DataLoader


# In[79]:


num_classes=2


# In[90]:


from typing import Optional

from torch import Tensor

from torch_geometric.utils import scatter

def global_add_pool_custom(x: Tensor, batch: Optional[Tensor],
                    size: Optional[int] = None, roots_to_embed:Optional[Tensor]=None, reduce_type='sum') -> Tensor:
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n.

    Functional method of the
    :class:`~torch_geometric.nn.aggr.SumAggregation` module.

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (torch.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    dim = -1 if x.dim() == 1 else -2

    if batch is None:
        return x.sum(dim=dim, keepdim=x.dim() <= 2)
    
    
    size = int(batch.max().item() + 1) if size is None else size
 
    
    if roots_to_embed is not None:
        
        ind_nonroot_long = roots_to_embed.to(torch.long)
        cond = torch.where(ind_nonroot_long ==1, True, False)#, 1.0, 0.0)
        # print('ind_nonroot_long', ind_nonroot_long.sum())
        
        x=x[cond]
        batch = batch[cond]
        
        # print('x ss',  x.shape)
        # print('batch', batch)
    
    # return
    # print('reduce_type', reduce_type)
    return scatter(x, batch, dim=dim, dim_size=size, reduce=reduce_type)


from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
# from torch_geometric.nn import global_add_pool, global_mean_pool
from gnns_no_feat import GNNNoEdge as GCNNoEdge

class GCN(torch.nn.Module):
    def __init__(self, reduce_type= 'sum' ,JK='last', net_norm='none', hidden_channels=128, num_layer=3,gnn_type='GATConv',drop_ratio=0.6 ):
        super(GCN, self).__init__()
        # torch.manual_seed(2)
        self.drop_ratio = drop_ratio
        self.reduce_type = reduce_type
        
        self.node_embed = GCNNoEdge(JK=JK, net_norm=net_norm,emb_dim = hidden_channels,
                             num_layer= num_layer, gnn_type=gnn_type,drop_ratio=drop_ratio).to(device)

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index,edge_attr,  batch, roots_to_embed=None):
        # def forward(self, x, edge_index, batch, roots_to_embed=None):
        x = self.node_embed( x, edge_index,edge_attr, batch)
        # 1. Obtain node embeddings 

        # 2. Readout layer
        # x = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        x = global_add_pool_custom(x,batch, roots_to_embed=roots_to_embed)
        # 3. Apply a final classifier
        x = F.dropout(x, p=self.drop_ratio, training=self.training)
        x = self.lin(x)
        
        return x

# model = GCN(hidden_channels=64)
# print(model)


# In[91]:


def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         # print('data.edge ', d"ata.x, data.edge_index)
         # break
        
         data=data.to(device)
         # print('data' , data.y)
         global gtemp
         gtemp = data
         # print('data', data.x.shape)
         # datax_rootonly = data.x*gtemp.roots_to_embed.view(-1,1)
         # print('datax_rootonly ', datax_rootonly.shape)
        
         out = model(data.x,  data.edge_index, data.edge_attr, data.batch)#, data.roots_to_embed)##None)#:data.roots_to_embed)  # Perform a single forward pass.
         # out = model(datax_rootonly,  data.edge_index, data.batch, data.roots_to_embed) 
         # break
         # print('out', out)
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.
         # print('loss', loss)

def test(loader):
     # print('test')
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         data=data.to(device)
         # print('data', data, data.x, data.edge_index)
         data = data.to(device)
         out = model(data.x, data.edge_index,data.edge_attr, data.batch, None)  
         
         out = F.softmax(out, dim=1)
         # print('out', out[-10:])
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         # correct += int((pred == data.y).sum())  # Check against ground-truth labels.
         pred = out.argmax(dim=1)  # Use the class with highest probability.

            
         score = roc_auc_score(data.y.to('cpu'), out[:,1].detach().to('cpu').numpy())
         return score



def test_our(loader):
     # print('tour')
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         # print('data', data, data.x, data.edge_index)
         data = data.to(device)
         out = model(data.x, data.edge_index,data.edge_attr, data.batch, data.roots_to_embed)  
         out = F.softmax(out, dim=1)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         score = roc_auc_score(data.y.to('cpu'), out[:,1].detach().to('cpu').numpy())
         return score
            


# In[89]:

from heapq import nlargest
from torch_geometric.loader import DataLoader

gnn_type = 'GATConv'

if gnn_type=='GATConv':
    JK='last'
    reduce_type='sum'
    net_norm='none'
    hidden_channels=64
    num_layer=3
    drop_ratio=0.2
    lr=0.001
    weight_decay =0.1

if gnn_type=='GCNConv':
    JK='last'
    reduce_type='sum'
    net_norm='none'
    hidden_channels=64
    num_layer=3
    drop_ratio=0.2
    lr=0.001
    weight_decay =0.1
    

if gnn_type=='GINConv':
    JK='last'
    reduce_type='sum'
    net_norm='none'
    hidden_channels=64
    num_layer=3
    drop_ratio=0.2
    lr=0.001
    weight_decay =0.1
    


num_steps=10
step_size = -1
decrement = [1,1]


for gnn_type in ['GATConv', 'GCNConv', 'GINConv']:

    threshold_class_dict  = {0:14
                             ,1:15
                            }


    threshold_array = [threshold_class_dict[0], threshold_class_dict[1]]


    #top 5 seed
    use_seed = [1,2,3,4,5,6,7,8,9,10]
    best_val_auc_across_step = -10000
    best_test_auc_at_best_val_across_step = -10000
    for step in range(num_steps): 


        threshold_array[0] = threshold_array[0]+decrement[0]*step_size
        threshold_array[1] = threshold_array[1]+decrement[1]*step_size

        print('threshold_array' , threshold_array)
        patterns_dict = {}

        for classid, transactions in transactions_array_classwise.items():
            print('classid', classid)
            patterns_dict[classid] = pyfpgrowth.find_frequent_patterns(transactions, threshold_array[classid])

            for pattern, freq in patterns_dict[classid].items():
                patterns_dict[classid][pattern] = freq#*1.0/len(transactions_array_classwise[classid])


        patterns_dict_temp  = {}

        for classid, pattern_freq in patterns_dict.items():
            patterns_dict_temp[classid] =  {}

            for pattern, freq in pattern_freq.items():
                patterns_dict_temp[classid][tuple(set(pattern))] = freq# freq*1.0/len(transactions_array_classwise[classid])


        patterns_dict = patterns_dict_temp

        p1_keys = list(patterns_dict[1].keys())
        p0_keys = list(patterns_dict[0].keys())
        p0_p1 = p1_keys + p0_keys
        print('len(set(p1_keys).difference(p0_keys)) ', len(set(p1_keys).difference(p0_keys)))
        print('len(set(p0_keys).difference(p1_keys)) ', len(set(p0_keys).difference(p1_keys)))
        print('p0_p1 ', len(p0_p1))

        graphs_py = []


        for classid, pattern_freq in patterns_dict.items():
            # print(classid,pattern_freq)
            print('classid', classid)

            count=0
            for pattern, freq in pattern_freq.items():
                Graph_combined = None

                #binary only
                if pattern in patterns_dict[1 - classid]:
                    continue

                # print ('pattern ', pattern)
                for label in pattern:

                    mptree = map_label_to_mptree[label]
                    if Graph_combined is None:
                        Graph_combined = mptree

                    else:


                        Graph_combined = nx.disjoint_union(Graph_combined, mptree)#, rename=("-{}-".format(patternCounter), "H"))


                pyg_graph = from_networkx(Graph_combined,group_edge_attrs=['label'])#map_label_to_mptree[1712120212021703030])
                # group_edge_attrs=['label']
                pyg_graph = pyg_graph.to(device)
                # print('classid ins', classid)
                pyg_graph.y = torch.tensor(classid).to(device)
                pyg_graph.x = pyg_graph.label
                #molbace
                pyg_graph.x = torch.tensor([node_labels_rev[x.item()] for x in pyg_graph.label]).long().to(device)
                pyg_graph.edge_attr = torch.tensor([edge_labels_rev[edge_attr.item()] for edge_attr in pyg_graph.edge_attr]).long().to(device)

                roots_to_embed = torch.zeros(pyg_graph.num_nodes)
                indices_root = torch.tensor(list(set(pyg_graph.edge_index[1].tolist()).difference(pyg_graph.edge_index[0].tolist()))).to(device)

                roots_to_embed[indices_root] = 1
                roots_to_embed=roots_to_embed.to(device)
                pyg_graph.roots_to_embed = roots_to_embed

                pyg_graph = pyg_graph.to(device)

                graphs_py.append(copy.deepcopy(pyg_graph))
                count+=1
            print('count', count)

        random.shuffle(graphs_py)
        train_dataset = graphs_py

        for a in train_dataset:
            a.x=a.x.long().to(device)
            a.edge_attr=a.edge_attr.long().to(device)    


        a_1 = []
        a_2 = []
        for x in train_dataset:
            # print(data.y)
            if x.y == 1:
                a_1.append(x.y)
            else:
                a_2.append(x.y)
        print('train_dataset ' , len(a_1),len(a_2))


        num_nodes=0
        num_edges=0
        for train_data_obj in train_dataset:
            e_i, e_attr, x_attr = train_data_obj.edge_index, train_data_obj.edge_attr, train_data_obj.x
            num_features = x_attr.shape[1]
            num_nodes+= x_attr.shape[0]
            num_edges+= e_i.shape[1]
            edge_dim = e_attr.shape[1]




        print('NOW train_dataset', len(train_dataset))
        print('val_dataset', len(val_dataset))
        print('test dataset', len(test_dataset))

        

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        train_loader_full_batch = DataLoader(train_dataset, batch_size=10000, shuffle=True)
        actual_orig_dataset_loader = DataLoader(actual_orig_dataset, batch_size=6000, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=6000, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=6000, shuffle=True)



        runs_stats_val_auc = []
        runs_stats_test_auc = []

        # top 5 seed

        for seed in use_seed:

            print('seed ', seed)
            seed_everything(seed)


            model = GCN(reduce_type= reduce_type , JK=JK, net_norm =net_norm, hidden_channels=hidden_channels, num_layer=num_layer,
                        gnn_type=gnn_type,drop_ratio=drop_ratio).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss()


            best_val_acc = -1000
            best_test = -1000
            train_our_acc = -1000
            train_acc=-1000
            val_acc= -1000
            best_test =-1000
            test_acc=-1000

            for epoch in range(0, 1500):

                # if (epoch+1) %1==0:
                #     train_our_acc = test_our(train_loader_full_batch)

                if epoch%10==0:
                    train_acc = test(actual_orig_dataset_loader)
                    val_acc = test(val_loader)

                    test_acc = test(test_loader)

                    if best_val_acc <= val_acc:
                        best_val_acc = val_acc
                        best_test = test_acc

                # print(f'Epoch: {epoch:},  val_acc: {val_acc:.4f}, Running Test Acc: {test_acc:.4f} , Actual test acc: {best_test:.4f} ')
                if epoch%100==0:
                    print(f'seed {seed } Epoch: {epoch:},train_our_acc {train_our_acc:.4f}  Train Acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, Running Test Acc: {test_acc:.4f} , Actual test acc: {best_test:.4f} ')

                    # print('best_test_auc_at_best_val_across_step ', best_test_auc_at_best_val_across_step , '  best_val_auc_across_step  ', best_val_auc_across_step)
                train()
            runs_stats_val_auc.append(best_val_acc)
            runs_stats_test_auc.append(best_test)

        runs_stats_val_auc =np.array(runs_stats_val_auc)
        runs_stats_test_auc = np.array(runs_stats_test_auc)

        nlargest_val = nlargest(5, runs_stats_val_auc)
        # print('nlargest_val ', nlargest_val)

        current_step_best_val = np.mean(nlargest(5, runs_stats_val_auc))
        current_step_best_val_index = [np.where(runs_stats_val_auc==i) for i in nlargest_val]
        # print('current_step_best_val_index ', current_step_best_val_index)

        current_step_best_test = np.mean(runs_stats_test_auc[current_step_best_val_index])


        if current_step_best_val > best_val_auc_across_step:
            best_val_auc_across_step = current_step_best_val
            best_test_auc_at_best_val_across_step = current_step_best_test


        print('\n\n\ RESULTS\n ::::::threshold_array ', threshold_array)
        print(' gnn_type ', gnn_type)

        print("\n ----- SIZE IS_--------")
        print('num edges', num_edges)
        print('num nodes', num_nodes)

        print('num_node_features', num_features)
        print('nodes into features', num_nodes*num_features)
        print('edge_dim', edge_dim)
        print('edges into edge features', num_edges*edge_dim)

        print('actual_orig_dataset ', len(actual_orig_dataset))


        print('** current_step_best_val ', current_step_best_val)

        print('** current_step_best_test ', current_step_best_test)
        print( '** best_test_auc_at_best_val_across_step ', best_test_auc_at_best_val_across_step)
        print('\n\n\n')

