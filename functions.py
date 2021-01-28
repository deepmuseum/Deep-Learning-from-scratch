import numpy as np

def conv2d(input,weight, bias, stride, padding=0, padding_mode='constant'):
    #https://cs231n.github.io/convolutional-networks/
    n_weight, h_weight, w_weight, c_weight=weight.shape
    h_input, w_input, c_input = input.shape
    input=np.pad(input,pad_width=((padding,padding),(padding,padding),(0,0)),mode=padding_mode, constant_values=0)
    h_input, w_input, _ = input.shape
    assert c_weight==c_input
    assert len(bias)==n_weight
    assert (w_input-w_weight+2*padding)%stride==0 and (h_input-h_weight+2*padding)%stride==0
    output = np.zeros(((w_input-w_weight+2*padding)//stride+1,(h_input-h_weight+2*padding)//stride+1,n_weight))
    for i_weight in range(n_weight):
        filter = weight[i_weight]
        for i in range(h_weight//2,h_input-h_weight//2,stride):
            for j in range(w_weight//2,w_input-w_weight//2,stride):
                output[(i-h_weight//2)//stride,(j-w_weight//2)//stride,i_weight]=np.sum(filter*input[i-h_weight//2:i-h_weight//2+h_weight,j-w_weight//2:j-w_weight//2+w_weight])+bias[i_weight]
    return output


if __name__=="__main__":
    input=np.random.normal(0,1,(7,7,3))
    weight=np.random.normal(0,1,(2,3,3,3))
    bias=np.zeros(2)
    output=conv2d(input,weight,bias,2)
    print(output)