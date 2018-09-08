# ShuffleNetV2_tensorflow
a tensorflow based implementation of ShuffleNetV2 on the Tiny ImageNet dataset

## File Organization
ShuffleNetV2  
|---data: tiny imagenet dataset  
|---|---test  
|---|---train  
|---|---val  
|---model: save checkpoint file  
|---|---tensorboard: save tensorboard file  
|---src: source codes  
|---|---main.py: data load, model training and test functions  
|---|---model.py: ShuffleNetV2 model  
|---rst: result on the test set  
##  Description
The model use the 0.5x weights configure.  
I trained the model from the scratch on the [Tiny ImageNet dataset](http://tiny-imagenet.herokuapp.com/).  
I trained two rounds.  
On the first round, the learning rate was set as 0.5 and tricks such as warm up and exponential decay were used.  
It trained 150 epochs. The accuracy on the validation set achieved above 80%.  
On the second round, the learning rate was set as 0.0005 and exponential decay was used.  
It trained 50 epochs. The accuracy on the validation set achieved about 90%.  
I tested on the test dataset. But I failed to get the accuracy because the website crashed after I uploaded.
