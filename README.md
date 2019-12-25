## Face recognition by Radial Basis Function Network (RBFN)

This project demonstrates face recognition using principal component analysis (PCA) and
radial basis function network (RBFN). More specifically, principal component analysis has been used for feature extraction and radial basis
function network as a classifier. 

#### Experimental setup
In this project AT&T face database (or ORL face database) has been used. It contains 400 grayscale images of 40 persons. Each person has 10 images, each having a resolution of 112x92, and 256 gray levels.

Training images are stored in 40 different folders (s1, s2, ...., s40). Each folder is for one signle person
and contains 5 images of that person. Similarly, there are 40 test folders (t1, t2, ...., t40).
Each test folder contains 5 test images of an individual. 

###### Exp 01
In the first experiment we took 200 images of 40 different persons. Each person has 5 images. Then we trained with them. After that, we added different levels of noise with the 200 images and took as test images.

###### Exp 02
In this experiment 5 images of an individual are kept for training and another 5 images of the same individual are kept for testing. Thus, we got a total of 200 images of 40 different persons in our training data set and 200 images in our testing data set.

###### Exp 03
In this experiment we took 200 images of 40 different persons for training. The remaining 200 images were added by different levels of noise and were taken as testing data set.

###### Exp 04
In this experiment, we took 10 images of a single person and trained with them. An image is randomly chosen from the 10 images. We then added noise to this image and took as test image. 

#### Reference
Dhar, Mrinal Kanti, Quazi M. Hasibul Haque, and Md Tanjimuddin. "Face Recognition by Radial Basis Function Network (RBFN)." International Journal of Computer Applications 78.3 (2013): 0975-8887. 