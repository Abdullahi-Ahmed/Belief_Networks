# Introduction to Neural Networks.  
Artificial neural networks is a big part of our personal and work life today, i.e youtube recommending a new song or Amazon suggesting a compliment product e.t.c, But Where did it all start?, why is it Resurgence NOW? And What is behind this mask?  
### Where did it all start?
In a simplist form neural networks mimics the way human brain operates. the ability to learn, adopt in a changing  enviroment, analyze incomplete  and  unclear,  fuzzy  information, and  make  its  own judgment using circuit of neurons  
![The-evolution-of-neural-networks](https://user-images.githubusercontent.com/85021780/145255399-3389375b-7b4a-4b25-befc-68ee7a2b5956.png)  
Why is Artificial intelligence popular Now while it date back decades?  

**Big Data** - According to Forbes "There are 2.5 quintillion bytes of data created each day at our current pace, but that pace is only accelerating with the growth of the Internet of Things (IoT). Over the last two years alone 90% of the data in the world was generated."  
huge data generated today and the easier collection & storage make neural networks resurgence to learn and productize into Appilications.  

**Hardware** - Masive parallelizable and the intoduction of GPUs for Computational processing make it ease for implementation .   

**Software** - Continuos improvement in Technology, new Models & Algorithms with the help of Toolboxes like TensorFlow makes neural network Appilication easier and popular.   

### Behind the scenes
Neural Networks is a network or circuit of neurons. Perceptron is single neuron and the structural building block of deep learning. Is it very important to understand how a neuron works to make predictions/judgment.  
<img width="509" alt="Screenshot 2021-12-08 213421" src="https://user-images.githubusercontent.com/85021780/145265199-8f0e9ca3-db34-4af5-a61f-88409cdf8001.png">
  
Above: To get the output (prediction) We Multiply each input data (x1, x2 ...... xm) with it's Corresponding Weight (w1, w2 ....... wm) and then add all together including (bias x it's weight). We Again take the result of the single added sum and Pass through non - linear activation Function.   
        
    Y = g(bias.w0 + (x1.w1 + x2.w2 ..... + xm.wm)
    
    Y = Output
    g = Activation Function
    w = weights
    x = inputs  

**Inputs** - are the data points fed through the network to learn and make the prediction.  
**weights** - is the parameter within a neural network that transforms input data within the network and decides how much influence the input will have on the output.  
**Bias** - this allows you to shift your activation function  from left or right.   
**Activation Function** - decides whether a neuron should be activated or not. This means that it will decide whether the neuron's input to the network is important or not in the process of prediction using simpler mathematical operations.  
<img width="509" alt="Screenshot 2021-12-08 215343" src="https://user-images.githubusercontent.com/85021780/145267011-67c0d1e7-b61f-4f27-a887-48e5b1e8ec24.png">  
In the left-diagram imagine you're working on binary classification problem, To separate the two classes (red & green) using linear activation function to make the decision boundary no matter how deep the neural network is you'll not get a good prediction unless (apply the right-diagram) by pass through Non-linear activation fuction to make a decision boundary that's much accurate. there many activation function you can choose from i.e Sigmoid , hyperbolic tangent, ReLU and Softmax.  
  
## Now let us understand forward propagation with the help of an example (TO predict if a client will buy a product )
  

    
    let's predict Y (if client will buy a product) given:
    X1 = number of visit to the store
    X2 = price of the product
    X0 = bias value
    W  = Corresponding WeightS
    g = Activation Fuction (sigmoid)
    
    X1 = 2 ,  W1 = -0.49
    X2 = $0.88 ,  W2 = 0.97
    X0 = 0 , W0 = 1
    
    Y = g ( 2 x -0.49 + 0.88 x 0.97 + 0 x 1 )
    g = 1 / 1 + 2.71828183 ^ 0.1264
      = 0.46%
   
There 46% chance of the client to buy the product when we used only one neuron for forward propagation. but we can still use single layer neural network to make the prediction while applying the same formula.  
    
    g = Activation Fuction
    z=  linear combination  
<img width="509" alt="Screenshot 2021-12-08 235958" src="https://user-images.githubusercontent.com/85021780/145283649-658f1761-c351-48d7-990b-5e6f83494356.png">  
we compute the same formula in each neuron. here we take W1 as weights Corresponding to each input in the first layer and W2 as weights Corresponding to each input in the second layer.  
Lets take a look at one zoomed neuron (Z2)  
<img width="508" alt="Screenshot 2021-12-09 000458" src="https://user-images.githubusercontent.com/85021780/145284211-2801157b-b05e-4680-bd56-ebaf7e593d07.png">  
Here we're just computing neuron (Z2) from the network
     
     Z2 = (X1 * W1,2 + X2 * W2,2 + Xm * Wm,2)
     g(Z2)
We compute all the above throughout the network to predict Y.  
<img width="509" alt="Screenshot 2021-12-09 001908" src="https://user-images.githubusercontent.com/85021780/145286057-d9076756-f195-416a-9ac1-7ee27cbe71cf.png">  
To create create a deep neural network all we have to do is keep stacking these layers to create more and more hierarchical models to compute the final output by going deeper and deeper into the network while applying activation fuction in each linear combination of inputs in every possible combination of each layer in the network.  
## Further Readings
https://www.youtube.com/watch?v=5tvmMX8r_OM&t=1575s  
https://www.inf.ed.ac.uk/teaching/courses/nlu/assets/reading/Gurney_et_al.pdf  
https://medium.com/analytics-vidhya/what-do-you-mean-by-forward-propagation-in-ann-9a89c80dac1b

 

    






