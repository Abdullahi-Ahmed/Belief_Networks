# Introduction to Neural Networks.  
Artificial neural networks is a big part of our personal and work life today, i.e youtube recommending a new song or Amazon suggesting a compliment product, But Where did it all start?, What is behind this mask? And, why is it Resurgence NOW?  
### Where did it all start?
In a simplist form neural networks mimics the way human brain operates. the ability to learn, adopt in a changing a enviroment, analyze incomplete  and  unclear,  fuzzy  information, and  make  its  own judgment  out  of  it using circuit of neurons  
![The-evolution-of-neural-networks](https://user-images.githubusercontent.com/85021780/145255399-3389375b-7b4a-4b25-befc-68ee7a2b5956.png)  
Why is Artificial intelligence popular Now while neural networks date back decades?  

**Big Data** - According to Forbes "There are 2.5 quintillion bytes of data created each day at our current pace, but that pace is only accelerating with the growth of the Internet of Things (IoT). Over the last two years alone 90% of the data in the world was generated."  
huge data generated today and the easier collection & storage make neural networks resurgence to learn and productize into Appilications.  

**Hardware** - Masive parallelizable and the intoduction of GPUs for Computational processing ease the implementation of Artificial intelligence.   

**Software** - Continuos improvement in Technology, new Models & Algorithms with the help of Toolboxes like TensorFlow makes neural network Appilication easier and popular.   

### Behind the scenes
Neural Networks is a network or circuit of neurons also known as Perceptrons.  
Perceptron is single neuran and the structural building block of deep learning. Is it very important to understand how a neuron works to contribute to an output.  
<img width="509" alt="Screenshot 2021-12-08 213421" src="https://user-images.githubusercontent.com/85021780/145265199-8f0e9ca3-db34-4af5-a61f-88409cdf8001.png">
  
To get the output (prediction) We Multiply each iput data (x1, x2 ...... xm) with it's Corresponding Weight (w1, w2 ....... wm) and then add all together including (bias x it's weight). We take the result of the single added sum and Pass through non - linear activation Fuction.   
        
    Y = g(bias.w0 + (x1.w1 + x2.w2 ..... + xm.wm)
    
    Y = Output
    g = Activation Function
    w = weights
    x = inputs  

**Inputs** - are the data points fed through the network to learn and make the prediction. **weights** - is the parameter within a neural network that transforms input data within the network's hidden layers and decides how much influence the input will have on the output. **Bias** - this allows you to shift your activation function  from left or right.   
**Activation Function** - decides whether a neuron should be activated or not. This means that it will decide whether the neuron's input to the network is important or not in the process of prediction using simpler mathematical operations.  
<img width="509" alt="Screenshot 2021-12-08 215343" src="https://user-images.githubusercontent.com/85021780/145267011-67c0d1e7-b61f-4f27-a887-48e5b1e8ec24.png">  
In the left-diagram 






