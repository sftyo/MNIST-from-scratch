# MNIST-from-scratch
This project was inspired by [Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU&t=545s).

For the code, you can check it out [here](https://www.kaggle.com/code/setyoab/mnist-from-scratch).

## Explanation
MNIST dataset is a dataset that consist of black and white images of digit  starting from 0 to 9. In this project, I am creating a MNIST classifier using neural network from scratch (only numpy).

The images is a 28 x 28 pixel, which we will have 784 units in the input layer. The model architecture will consist of a input layer, hidden layer (ReLU), and output layer (Softmax). So we will have 784 units as the inputs, and it will outputs 10 units (into the ReLU)

```math
\begin{aligned}
X^{[0]} &: \text{Input matrix/vectors (784 x m)} \\ 
W^{[1]} &: \text{Weight for the input layer (10 x 784)} \\
b^{[1]} &: \text{Bias for the first hidden layer (ReLU) (10 x 1)} \\
W^{[2]} &: \text{Weight for the second layer (ReLU) (10 x 10)} \\
b^{[2]} &: \text{Bias for the second layer (10 x 1)}
\end{aligned}
```
So, in general we have this equation
```math
\begin{aligned}
	Z^{[1]} &= W^{[1]}X^{[0]} + b^{[1]} \quad \text{(10 x m)} \\
	A^{[1]} &= ReLU(Z^{[1]}) \quad \text{(10 x m)} \\
	Z^{[2]} &= W^{[2]}A^{[1]} + b^{[2]} \quad \text{(10 x m)} \\
	A^{[2]} &= Softmax(Z^{[2]}) \quad \text{(10 x m)}
\end{aligned}
```
Next step is to find the derivative of the error in respect to $W^{[1]}, b^{[1]} , W^{[2]}, b^{[2]}$  in order to implement gradient descent (updating parameters).  For easier derivatives and math manipulation, we are going to use 'error' function as our loss function, but in order to do that, our output will be a one hot encode of the true labels. For example if $y = 4$, then it will become $\vec{y} = [0, 0, 0, 1, \cdots, 0]$ , thus the loss function will be
```math
E = \dfrac{1}{2m}\sum_{i = 1}^{m}(A^{[2]} - \vec{y})^2
```
Then by using chain rule we have:
```math
\begin{align}

dW^{[2]} &= \dfrac{\partial E}{\partial W^{[2]}} = \dfrac{\partial E}{\partial A^{[2]}} \cdot \dfrac{\partial A^{[2]}}{\partial Z^{[2]}} \cdot \dfrac{\partial Z^{[2]}}{\partial W^{[2]}} \\

db^{[2]} &= \dfrac{\partial E}{\partial b^{[2]}} = \dfrac{\partial E}{\partial A^{[2]}} \cdot \dfrac{\partial A^{[2]}}{\partial Z^{[2]}} \cdot \dfrac{\partial Z^{[2]}}{\partial b^{[2]}} \\

dW^{[1]} &= \dfrac{\partial E}{\partial W^{[1]}} = \dfrac{\partial E}{\partial A^{[2]}} \cdot \dfrac{\partial A^{[2]}}{\partial Z^{[2]}} \cdot \dfrac{\partial Z^{[2]}}{\partial A^{[1]}} \cdot \dfrac{\partial A^{[1]}}{\partial Z^{[1]}} \cdot \dfrac{\partial Z^{[1]}}{\partial W^{[1]}} \\


db^{[1]} &= \dfrac{\partial E}{\partial b^{[1]}} = \dfrac{\partial E}{\partial A^{[2]}} \cdot \dfrac{\partial A^{[2]}}{\partial Z^{[2]}} \cdot \dfrac{\partial Z^{[2]}}{\partial A^{[1]}} \cdot \dfrac{\partial A^{[1]}}{\partial Z^{[1]}} \cdot \dfrac{\partial Z^{[1]}}{\partial b^{[1]}} \\
\end{align}
```
After we have the derivative of each parameters, we can update the parameters using gradient descent, that is:
```math
\begin{align}

W^{[1]} &= W^{[1]} - \alpha \cdot dW^{[1]} \\
b^{[1]} &= b^{[1]} - \alpha \cdot db^{[1]} \\ 
W^{[2]} &= W^{[2]} - \alpha \cdot dW^{[2]} \\
b^{[2]} &= b^{[2]} - \alpha \cdot db^{[2]} \\ 

\end{align}
```
where $\alpha$ is called the learning rate.
