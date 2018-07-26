# SDI
Our Project "SDI" - Skin Disease Identifier is based on Machine Learning trained with CNN (Neural Network) which solves the problem of visiting to doctors for skin checkup , anyone , irrespective of gender, age, and color can pass the image of effected part of skin  to our code and can get the best results among the melanoma,nevus and seborrheic keratosis skin diseases. It does not require any features to pass on , which makes it user friendly. It effectively gives the accurate results (~=70%). It is useful for the household women and rural area people.

Main: It is the main function in which all other dependent functions are called.

sdi_initial: It initializes the values for the filteration matrix used in CNNs.

sdi_train: In this function we call 2 different functions for forward propagation and back propagation as "sdi_for" and "sdi_back" respectively.

sdi_users: This function is made for users to pass on the image and get the output. for eg: sdi_users("test_melanoma(1).jpg");
We have provided few images for manual testing .

You will get probabely 60-70% accurate chances of the pedicted disease.

Important points:
sdi_train takes 1-2 hrs to fully train depending upon number and resolution of images.
sdi_users takes few seconds to predict the results.
