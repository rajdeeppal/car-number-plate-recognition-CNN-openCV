# car-number-plate-recognition-CNN-openCV
# INSTALL
This project requires Python 2.7 and the following Python libraries installed:

    NumPy
    keras
    umutils
    pickle
    Pandas
    matplotlib
    scikit-learn
    

You will also need to have software installed to run and execute a Jupyter Notebook

If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer.
# CODE
  In order to run the program to have build your datsets , to do that run create_training_and_test_data.py file . To file will save for store the image data and other for lables data data.pickle and labels.pickle . Run train_cnn.py to build the model . Now time for prediction , run predict.py before it run digit_crop.py this helps to segmented the number plate image (28 x 28 ) into crop images of digits of the number plate , then we can directly pass the images into our model for final prediction .
# RUN
  In a terminal or command window, navigate to the top-level project directory car-number-plate-recognition/

predict.py

# DATA
Dataset contains images of 0 - 9 and letter A - Z . For download the whole dataset link - https://drive.google.com/file/d/1dS0TVtmgzxjjarEKm23H9wNCzG-v1wz3/view?usp=sharing 
