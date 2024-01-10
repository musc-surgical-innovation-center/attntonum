# AttnToNum

## Overview
This research presents AttnToNum, an attention-based embedding technique for numerical values in text. The embeddings are derived from both number magnitude and context words. AttnToNum boosted performance for prediction of the composite outcome of Morbitidity or Mortality (MM) after a CABG (coronary artery bypass grafting) surgery. The input was the most recent pre-surgery clinical note.  

**Summary**  
AttentionToNum outperforms basic embeddings if and only if both the numbers are scaled first, and context is incorporated via the attention mechanism.
This implies that both scaling (magnitude) and attention (context) are important for capturing numerical information.  

See [example](./directions.ipynb) for how to incorporate these embeddings in your own analysis.


## ScaleNum
AttnToNum first scales the number through another embedding technique we present, ScaleNum, via a learned log->linear->sigmoid function. 
$$\vec{y}=σ(\vec{a}log(x)+\vec{b})=\frac{x^{\vec{a}}e^{\vec{b}}}{1+x^{\vec{a}}e^{\vec{b}}}$$

AttnToNum then updates the scaled number embeddings based on local attention.

The steps are as follows for CABG analysis:  
Remove out of context numbers (ROOC) -> Embed -> Scale Numbers -> Attention to the Number's Context  
    
ROOC is optional. Furthermore, residual connections between embedding layers can stabilize performance. See [example](./directions.ipynb) on simulated data for residual connections and no ROOC.  
   
In the image below, the embedded values correspond to the first dimension of the dimension-50 embeddings.  

![pipeline](fig4.png)


## How To Use (Reproducibility)
Due to the sensitive PHI nature of the data, we do not release the CABG notes dataset. A sample with simulated data is located [here](./directions.ipynb)  
  

## Considerations

Whether or not this embedding will boost classification performance on your dataset depends on key factors:

* Are the numbers predictive?
  * In our case, pre-operative BUN & creatinine values were predictive
* How many unique numbers are in the data?
  * The more unique numbers, the higher the likely performance boost with AttnToNum compared to standard tokenization and embedding. Though other options are available such as rounding the numbers to reduce the token space. With standard tokenization and embedding, the model does not incorporate numerical magnitude unless the same number was found in the train set.
* Negative numbers
  * ScaleNum first takes the log, which only accepts strictly positive numbers in it's domain. Therefore the vocab that tokenizes the text will need to convert negative numbers to positive. For example, clip the values to a very small positive number, or add an offset to negative numbers to make them positive. See line 151 of [vocabs.py](./vocabs.py).
  
