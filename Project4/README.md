# Architectural Design
For the Experimentations done with the fourth assignment, The following are the design principle considered in the same order.

  ## 1. Receptive field
  The first thing to take to consideration is to see manually at what size we can see patterns by manually zooming into the images that we have. for the MNIST dataset when manually zoomed it was found at 7*7 we are able to see meaningful patterns.
  ## 2. How Many Layers.
  The next thing while designing the cnn architecture is to decide about how many layers to use.It depends on the size of the image. Also it depends on the effective global receptive field of the dataset being used. For MNIST dataset as the size of object is not equal to size of the image, Global receptive field of 24*24 is enough.for a block in the architecture, order of increase and decrease of kernels in a block should be followed.Both the extremums of the number of kernels should be tested based on the inter and intra class variations of the dataset.The number of layers in a block depends on the receptive field where we find edges and gradients
 ## 3. 3*3 convolutions
 The 3 * 3 convolutions acts as feature extractors from the images . They are optimized to do convolutions faster and efficiently.As they are  odd they has line of symmetry. for mnist dataset the effective receptive field is around 7 * 7. I have placed 3 convolutions layers before  transition layers
 ## 4 MaxPooling
 The next thing to take into consideration is the placement of maxpooling layer in the network.Maxpooling takes care of the positional invariance of the feature present also a bit of rotational invarince. It should be placed after effective receptive field where edges and gradients,textures, patterns, parts of objects and objects are formed. By Maxpooling, We can considerable reduce the number of convolutional layers to be added to the network. In this experimental case, after 7 * 7 its observed that certain edges and gradients etc are formed so Placed maxpooling after the effective global receptive field of 7 * 7. 
 ## 5.Position of MaxPooling
 It should be placed after effective receptive field where edges and gradients,textures, patterns, parts of objects and objects are formed.In this experimental case, after 7 * 7 its observed that certain edges and gradients etc are formed so Placed maxpooling after the effective global receptive field of 7 * 7.
 ## 6. 1*1 convolutions
 1 * 1 convolutions are helpful in combining effectively  the spatially related channels so it is a way of decreasing the number of channels
 after convolutions. Hence it helps in reducing the number of parameters to be learnt. 1 * 1 is generally placed with maxpooling and hence both of them are together called Bottleneck layers. We also use 1 * 1 in the prediction layer to get the number of channels that should be equal to the number of classes.
 ## 7. Kernels and how do we decide the number of kernels?
 for a block in the architecture, order of increase and decrease of kernels in a block should be followed.Both the extremums of the number of kernels should be tested based on the inter and intra class variations of the dataset. if the intra and inter class variations are more 
 we will be going for higher kernels and if the variation is less we would prefer for low number of kernels.
 ## 8.Concept of Transition Layers
 Transition layers should be placed after certain number of convolutions which depends on the effective receptive field where edges and gradients,textures,patterns,parts of objects and objects are formed. they should not be placed near prediction layer which will cause more loss of information. for MNIST dataset I placed one transition layer after a global receptive field of 7 * 7.since the image size is of 28 * 28. In prediction block did not use any Transition layer.
 ## 9.Position of Transition Layer
 We decide number of blocks based on the size of the image. Number of layers in each block depends on the effective receptive field for that particular dataset where we find edges and gradients,textures,patterns,parts of objects and objects.transition layers consists of 1*1 convolution and maxpooling. we place the transition layers in all blocks except the last block where we have the prediciton layer. we use only 1 * 1 convolution in the  last block to get required number of classes from the channels present.
 ## 10. SoftMax
 Softmax is used as the last layer for classification problems. it gives probability like scores for the classes present. It should be 
 used based on the criticality of the problem statement. if the problem is critical like in the case of medical domain, we need to use the logits instead of softmax. for mnist I used softmax as the last layer in my network architecture.
 ## 11. Number of Epochs and when to increase them
 While doing experimenting if the training loss is higher than validation loss one of the reasons is might be we need to train the model 
 a bit more. so we need to increase the number of epochs.But too many epochs if trained it might lead to overfitting a sign when error rate increases.
 ## 12.When do we stop convolutions and go ahead with a larger kernel or some other alternative
 When the size of channels will be small around (7 * 7 or 11 * 11) if we do further convolutions the information may get diffused as the central pixels gets much attention than the surrounding pixels its the same case of a checkerboard issue, in that case we stop regular convolutions and go ahead with a larger kernel or some alternative like 1 * 1 convolution followed by global average pooling.
 ## 13. How do we know our network is not going well, comparatively, very early
 When we train the model within 1-2 epochs we can know our network is not going well when benchmarked with other network architecture for the same problem. if within 1-2 epochs our validation accuracy is much less than than other model, its a indication that we need to change our network.
 ## 14.When to add validation checks
 Its better to add validation checks while training provided that validatation check does not take much time. if it takes much time its better to do validation check after training for some epochs.
 ## 15. Image Normalization
 Images might bright, hazy etc. if they are not normalized before passing to cnn architecture Back propagation would try to figure out different kernels for the same feature extraction. But if the image is normalized, same single kernel would figure out the same feature 
 
 ## 16. Batch Normalization
 As Kernels are randomly initialized some kernels are scream very loudly if the feature is present and some kernels do scream very quietly even if the feature is found. so to we normalize all the channels by Batch Normalization so that each channel has its equal impact on the subsequent convolutions.
 ## 17. The distance of Batch Normalization from Prediction
 We do Batch Normalization after every convolution except for the prediction layer. We should not do Batch Normalization after the prediction layer as we need a clear cut differentiator in the final output to which class it belongs so we avoid doing batch normalization in the prediction layer.
 ## 18 . Learning Rate
 Our learning rate should be higher in the initial phase of training so we will converge faster. but as it goes near the optimal point where our loss would be minimum our learning rate should gradually decrease. Also our learning rate should be such that it should not get stuck in local minima.
 ## 19. DropOut
 We use Dropout as a Regularization technique. We do prefer using Data Augmentation and l2 reguralization normally. Nowadays using Dropout became obsolute.Dropout in cnn makes kernel the capability to extract multiple features which is not requires as kernels are designed to identify a single feature.
 ## 20.When do we introduce DropOut, or when do we know we have some overfitting
 We introduce Dropout when we dont have Data Augmentation for doing regularization to avoid overfitting. We can introduce dropout in Convolution layers or in transition layers or in both convolution and transition layers. we know we have some overfitting if our validation error rate gradually increase while training a few more epochs.
 ## 21. Batch Size, and effects of batch size
 Larger batch size helps in parallezing the training so it speeds up the training. Too much large Batch size may result in out of memory
 Error. Studies reveal that for a fixed learning rate,with increase in batch size validation accuracy increases and then decreases.
 ## 22.LR schedule and concept behind it
 A good LR scheduler should schedule learning rate should be higher in the initial phase of training so we will converge faster. but as it goes near the optimal point where our loss would be minimum our learning rate should gradually decrease. Also our learning rate should be such that it should not get stuck in local minima.
 ## 23. Adam vs SGD
 This is the last thing to try out but their impact might not have significant impact on the model.
 
 
 
 

 
 
 
 
