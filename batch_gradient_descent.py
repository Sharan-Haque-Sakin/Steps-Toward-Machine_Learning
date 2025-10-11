import numpy as np

def cost_function(X,y, theta): # The J(theta) function


    hypo = X.dot(theta) # h(x) = theta_transpose * X

    cost = np.sum(( y - hypo) ** 2) 

    return cost 


def grad_desc(X,y,theta,alpha,num_iters):

    J_history = [] # This will keep history of all the cost function value J(theta)
    hypo = X.dot(theta) 

    for i in range(num_iters):
        
        gradient= X.T.dot(hypo - y)

        theta = theta - alpha * gradient

        J_history.append(cost_function(X,y,theta)) 

    return theta,J_history

# Example

X = np.array(
    [[1,2],
    [3,4],
    [4,8]]
)

y = [5,7,9]

theta = np.array([0,0])

alpha = 0.001

num_iters = 100

print(grad_desc(X,y,theta,alpha,num_iters))