# DiRetNet---A-Deep-Convo-Net-for-determining-Diabetic-Retinopathy-category-
A deep Convolutional Neural Network made and trained using Keras

Inspired by the research done by the Google Brain team showcased in the TensorFlow Dev Summit 

The dataset used for this project is from the Kaggle competition on Determining Diabetic Retinopathy(0,1,2,3,4) from Retinal images  

https://www.kaggle.com/c/diabetic-retinopathy-detection

The concept and it's description has been specified in the video below 

P.S. - Thier objecitive was quite different. 

https://youtu.be/oOeZ7IgEN4o

Detailed information can be found in the documentation(Black_book.pdf) 

Currently, the model gives a 92% accuracy of the validation set, but there is a catch..

The dataset downloaded from kaggle was not clean and the model used to get confused while determining correct category for the Category 2 images. Hence, the final model trained and used is the one which outputs the predictions for 4 categories namely: 0, 1, 3 and 4. 

Futher work can be done and the dataset for category 2 can be included in the future if obtained from sources like hospitals and eye clinics.  
