import numpy as np


class relu():
    def __init__(self,):
        pass
    def forward(self,x):
        self.x = x
        self.x[self.x<0] = 0 # if x>0 return x else return 0
        return x
    def backward(self,x):
        self.x = x
        grad_rel = np.zeros_like(self.x)
        grad_rel[self.x>0] = 1.0
        return grad_rel
class softmax():
    def __init__(self,):
        pass
    def forward(self,x,axis):
        self.x = x
        self.axis = axis
        # return np.exp(x)/(np.exp(x).sum(axis=axis,keepdims = True))4
        self.out = np.exp(self.x - np.max(self.x)) / np.sum(np.exp(self.x - np.max(self.x)),axis=self.axis, keepdims =True)
        return self.out
    def backward(self,x):
        self.x = x
        N,dim = self.x.shape[0],self.x.shape[-1]

        sj = np.zeros((N,dim,dim))
        for i in range(dim):
            for j in range(dim):
                if i ==j:
                    sj[:,i, j] = self.x[:,i] * (1 - self.x[:,j])
                else:
                    sj[:,i, j] =  - self.x[:,i]*(self.x[:,j])
        return sj
def grad_w():
    pass
    # del_loc_w =
class model():
    def __init__(self,input ,output_dim,H = 2):
        self.input = np.array(input)
        print(self.input.shape)
        self.input = self.input.reshape(self.input.shape[0],-1) # (N,D)
        self.D = self.input.shape[-1]
        # self.D = input_dim
        self.H = H
        self.c= output_dim
        self.w1 = np.random.randn(self.D,self.H).T
        self.b1 = np.random.randn(self.H,).T
        self.w2 = np.random.randn(self.H,self.c).T
        self.b2 = np.random.randn(self.c,).T
        self.relu = relu()
        self.softmax =softmax()
    def forward(self):
        self.x = self.input #(2,10)
        self.x1 = self.x@self.w1.T # (2,5)
        print(self.x1.shape)
        self.x2 =self.x1 + self.b1 #(2,5)
        self.x3 = self.relu.forward(self.x2) # (2,5)
        self.x4 = self.x3@self.w2.T   #(2,10)
        self.x5 = self.x4 + self.b2 #(2,10)
        self.x6 = self.softmax.forward(self.x5,1) #(2,10)
        return self.x6
    def backward(self):
        # del_3 = np.zeros((10))
        # del_3[5] = 1.0
        # print(del_3.shape,self.x3.shape)
        # grad_x6_x5 = self.softmax.backward(self.x5) @ del_3    # (2,10,10)
        # print(grad_x6_x5.shape)
        # del_2 = del_3@self.w2
        # del_b2 = del_3
        # del_w2 = self.x3@del_3
        grad_l = np.zeros((2,10,1))
        grad_l[:,5,:] = 1.0 #del3 = []
        print(grad_l.shape)
        # grad_x6_x5 = np.transpose(self.softmax.backward(self.x5),axes=[0,2,1]).dot(grad_l) #(2,10,10)
        grad_x6_x5 = (self.softmax.backward(self.x5) * grad_l).sum(-1)
        print('ab',grad_x6_x5.shape)
        grad_x5_x4 = 1.0 * grad_x6_x5 #(2,10,10)
        grad_x5_b2 = 1.0 * grad_x6_x5 #(2,10,10)
        # print(grad_x5_x4@self.w2.T )
        grad_x4_x3 = grad_x5_x4.dot(self.w2)
        # grad_x4_w2 = np.matmul(self.x3,np.transpose(grad_x4_x3,axes=[0,2,1]),axis = 1)
        print('aa',np.transpose(grad_x4_x3, axes=[0, 1, 2]).shape)
        grad_x4_w2 = grad_x4_x3.dot(np.transpose(self.x3,axes=[1,0]))
        print('h',self.x3.shape ,grad_x4_w2.shape)
        grad_x3_x2 = self.relu.backward()@grad_x4_x3
        grad_x2_x1 = 1.0 * grad_x3_x2
        grad_x2_b1 = 1.0 * grad_x3_x2
        grad_x1_x  = self.w1@grad_x2_x1
        grad_x1_w1 = self.x @ grad_x2_x1

class model2():
    def __init__(self, input_dim, output_dim, H=2):
        # self.input = np.array(input)
        # print(self.input.shape)
        # self.input = self.input.reshape(self.input.shape[0], -1)
        # print(self.input.shape)# (N,D)
        self.D = input_dim
        # self.D = input_dim
        self.H = H
        self.c = output_dim
        self.w1 = np.random.randn(self.D, self.H)
        self.b1 = np.random.randn(self.H, )
        self.w2 = np.random.randn(self.H, self.c)
        self.b2 = np.random.randn(self.c, )
        self.relu = relu()
        self.softmax = softmax()

    def forward(self,x):
        self.z = np.dot(x,self.w1) #+ self.b1.T
        self.z2 = self.relu.forward(self.z)
        self.z3 = np.dot(self.z2,self.w2) #+ self.b2.T
        output = self.softmax.forward(self.z3,1)
        return output

    def backward(self,x,y,output):
        self.output_error = y - output
        self.output_delta = self.output_error*self.softmax.backward(output)

        self.z2_error_w2 = self.output_delta.dot(self.w2.T)
        # self.z2_error_b2 = 1.0
        self.z2_delta_w2 = self.z2_error_w2*self.relu.backward(self.z2)

        self.w1+=x.T.dot(self.z2_delta_w2)
        self.w2+=self.z2.T.dot(self.output_delta)
        # self.b1+=
        return None
if __name__ =='__main__':
    np.random.seed(1)
    # img = np.random.randn(2,3,3)
    # orig = np.random.randn(2,1,)
    img  = np.array(([2,0],[1,5],[3,6]),dtype = float)
    y = np.array(([92],[85],[59]),dtype = float)
    img = img/np.amax(img,axis=0)
    y = y/100
    mdl = model2(2,1,H=3)
    out = mdl.forward(img)
    mdl.backward(img,y,out)
    back = np.array([1])

    print(out.shape,back, back.shape)

