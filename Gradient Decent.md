## Gradient Decent

$\theta^{*}=\arg \underset{\theta}{\min} L(\theta) \quad$

L : loss function
$\theta_{i}^{k}:$ parameters( i 代表是第几个参数，k代表第几次迭代计算的结果)

Suppose that $\theta$ has two variables $\left\{\theta_{1}, \theta_{2}\right\}$ 

Randomly start at $\theta^{0}=\left[\begin{array}{l}{\theta_{1}^{0}} \\ {\theta_{2}^{0}}\end{array}\right] \quad$ 

$\theta$处的梯度gradient记作：$\nabla L(\theta)=\left[\begin{array}{l}{\partial L\left(\theta_{1}\right) / \partial \theta_{1}} \\ {\partial L\left(\theta_{2}\right) / \partial \theta_{2}}\end{array}\right]$

$\left[\begin{array}{l}{\theta_{1}^{1}} \\ {\theta_{2}^{1}}\end{array}\right]=\left[\begin{array}{l}{\theta_{1}^{0}} \\ {\theta_{2}^{0}}\end{array}\right]-\eta\left[\begin{array}{l}{\partial L\left(\theta_{1}^{0}\right) / \partial \theta_{1}} \\ {\partial L\left(\theta_{2}^{0}\right) / \partial \theta_{2}}\end{array}\right] \Rightarrow \theta^{1}=\theta^{0}-\eta \nabla L\left(\theta^{0}\right)$

$\left[\begin{array}{c}{\theta_{1}^{2}} \\ {\theta_{2}^{2}}\end{array}\right]=\left[\begin{array}{c}{\theta_{1}^{1}} \\ {\theta_{2}^{1}}\end{array}\right]-\eta\left[\begin{array}{c}{\partial L\left(\theta_{1}^{1}\right) / \partial \theta_{1}} \\ {\partial L\left(\theta_{2}^{1}\right) / \partial \theta_{2}}\end{array}\right] \Rightarrow \theta^{2}=\theta^{1}-\eta \nabla L\left(\theta^{1}\right)$

