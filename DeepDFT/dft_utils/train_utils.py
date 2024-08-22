import numpy
import mindspore as ms
import mindspore.nn as nn
import mindspore.common.dtype as mstype


def convert_batch_int32(batch:dict):
    batch32 = {}  
    for key, value in batch.items():  
        if isinstance(value, ms.Tensor) and value.dtype == mstype.int64:   
            converted_value = ms.Tensor(value.asnumpy().astype('int32'), mstype.int32)  
                    #key: [nodes, num_nodes, num_atom_edges, num_probe_edges, num_probes]
        else:   
            converted_value = value
        batch32[key] = converted_value
    return batch32

def flatten(lst:list):  
    # 用于将list(train_loader)转化成单层列表 --Carzit:我不是很明白
    result = []  
    for i in lst:  
        if isinstance(i, list):  
            result.extend(flatten(i))  
        else:  
            result.append(i)  
    return result

class CustomWithLossCell(nn.Cell):
   def __init__(self, backbone, loss_fn):
       super(CustomWithLossCell, self).__init__(auto_prefix=False)
       self._backbone = backbone
       self._loss_fn = loss_fn

   def construct(self, 
                 nodes, 
                 atom_edges, 
                 atom_edges_displacement, 
                 probe_edges, 
                 probe_edges_displacement, 
                 probe_target, 
                 num_nodes, 
                 num_atom_edges, 
                 num_probe_edges, 
                 num_probes, 
                 probe_xyz, 
                 atom_xyz, 
                 cell):
       out = self._backbone(nodes, atom_edges, atom_edges_displacement, probe_edges, probe_edges_displacement, num_nodes, num_atom_edges, num_probe_edges, num_probes, probe_xyz, atom_xyz, cell)
       return self._loss_fn(out, probe_target)

   @property
   def backbone_network(self):
       return self._backbone

            