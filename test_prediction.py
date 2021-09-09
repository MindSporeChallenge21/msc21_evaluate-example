

from participant_model import post_process

import mindspore as ms

import code


i = ms.Tensor([[
    [0,2,3,4],
    [0,2,3,4],
    [0,2,3,4],
    [0,2,3,4],
]])


j = ms.Tensor([[
    [0,2,0.5,4],
    [0,2,0.5,4],
    [0,2,0.5,4],
    [0,2,0.5,4],
]])

print(i.shape)

print(i.shape)

x = post_process(
    '', 
    (
        i,j,ms.Tensor([10,20])
        
    )
     
    
)
# enter interactive

code.interact(local=locals())

