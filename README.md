# minkf
Kalman filter/smoother, nothing more. A minimal implementation with only `numpy` dependency. Estimates the states of the system

<img src="https://latex.codecogs.com/svg.image?\begin{align*}x_{k}&space;&=&space;M_{k}&space;x_{k-1}&plus;u_k&plus;N(0,Q_k)&space;\\y_k&space;&=&space;K_k&space;x_k&space;&plus;&space;N(0,&space;R_k).&space;\end{align}&space;" title="\begin{align*}x_{k} &= M_{k} x_{k-1}+u_k+N(0,Q_k) \\y_k &= K_k x_k + N(0, R_k) \end{align} " />

Calculates also the likelihood of the data, in case one wants to do some hyperparameter tuning. One can also sample from the posterior distribution of the states.
