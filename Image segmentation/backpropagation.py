import numpy as np
import torch

x= np.array([[2., 3.]])
y= 1
w1= 0.11
w2= 0.21
w3= 0.12
w4= 0.08
w5= 0.14
w6= 0.15
W1= np.array([[w1, w3],[w2, w4]])
W2= np.array([[w5, w6]])

def forward(W1, W2):
    H= np.matmul(x, W1)
    H1= np.dot(x,W1)
    print("H:", H)
    #print("H1:", H1)
    output= float(np.matmul(H, np.transpose(W2)))
    #output2= float(np.dot(H, W2.T))
    print("output:", output)
    #print("output2:", output2)
    return H, output

def calc_error(output, real):
    return (1/2)*(output-real)**2

def backward(W1, W2, H, x, output, y):
    lr= 0.05
    delta= output-y
    W2n= W2-(lr*delta*H)
    #print("W2n:",W2n)
    W1n= W1- (np.matmul(lr*delta*np.transpose(x), W2))
    #print("W1nss:", W1n)
    return W1n, W2n

H, output= forward(W1, W2)

# for i in range(20):
#     print("loop:", i+1)
#     H, output= forward(W1, W2)
#     print("output:", output)
#     error= calc_error(output, y)
#     print("e:", error)
#     W1, W2= backward(W1, W2, H, x, output, y)
#     H, output= forward(W1, W2)

##https://hmkcode.com/ai/backpropagation-step-by-step/




# # N is batch size; D_in is input dimension;
# # H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 1, 2, 1, 1

# # Create random Tensors to hold inputs and outputs
# x = torch.tensor([2., 3.])
# y = torch.tensor([1.])

# print(x,y)
# # Use the nn package to define our model as a sequence of layers. nn.Sequential
# # is a Module which contains other Modules, and applies them in sequence to
# # produce its output. Each Linear Module computes output from input using a
# # linear function, and holds internal Tensors for its weight and bias.
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, D_out),
# )

# # The nn package also contains definitions of popular loss functions; in this
# # case we will use Mean Squared Error (MSE) as our loss function.
# loss_fn = torch.nn.MSELoss(reduction='sum')

# learning_rate = 0.05
# for t in range(10):
#     # Forward pass: compute predicted y by passing x to the model. Module objects
#     # override the __call__ operator so you can call them like functions. When
#     # doing so you pass a Tensor of input data to the Module and it produces
#     # a Tensor of output data.
#     y_pred = model(x)
#     print("y:",y_pred)
#     # Compute and print loss. We pass Tensors containing the predicted and true
#     # values of y, and the loss function returns a Tensor containing the
#     # loss.
#     loss = loss_fn(y_pred, y)
#     if t % 10 == 9:
#         print(t, loss.item())

#     # Zero the gradients before running the backward pass.
#     model.zero_grad()

#     # Backward pass: compute gradient of the loss with respect to all the learnable
#     # parameters of the model. Internally, the parameters of each Module are stored
#     # in Tensors with requires_grad=True, so this call will compute gradients for
#     # all learnable parameters in the model.
#     loss.backward()

#     # Update the weights using gradient descent. Each parameter is a Tensor, so
#     # we can access its gradients like we did before.
#     with torch.no_grad():
#         for param in model.parameters():
#             param -= learning_rate * param.grad
  