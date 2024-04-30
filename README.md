# TinyMLGesture
Gesture recognition technology has become a pivotal part of modern human-computer interaction, enabling devices to understand and interpret human gestures as commands. The objective of this project is to develop a gesture recognition system using a Raspberry Pi integrated with an accelerometer. This technology can serve in various applications such as virtual reality, automated systems, and as assistive technology for individuals with disabilities.

# Data Collection
For this project, a series of common and distinct gestures were selected, including
pointing, raising a hand, and other symbolic gestures. Data for each gesture was collected in ten trials using the MPU6050 accelerometer, which provides x, y, and z coordinates of motion. The data collection was facilitated by custom scripts utilizing the Matplotlib library for visualization and the Keyboard library for managing the recording process. The raw data was then processed to calculate the magnitude using the formula sqrt(x^2 + y^2 + z^2), ensuring consistency across trials.

# Data Collection - Flowchart
![TinyMLRecordingFlowchart-2](https://github.com/hydrenoid/TinyMLGesture/assets/82002017/d6f65101-6e6d-43f3-8a10-487f12a13d60)

# Data Preprocessing
Once collected, the data was loaded from CSV files into a Pandas dataframe for
manipulation and analysis. The preprocessing steps involved plotting the raw x, y, and z values for each trial and calculating statistical features such as mean, standard deviation, maximum, minimum, kurtosis, and skewness. These features were crucial for the next phase of model training, providing a robust dataset for effective learning outcomes.

# Model Training
The data was divided into feature sets and labels, with the features representing the calculated statistical values and the labels the corresponding gestures. The dataset was split using
K-folds validation, specifically five splits, to ensure the model’s generalizability. A Random
Forest Classifier was chosen due to its efficacy in handling non-linear data with multiple features. The model achieved an average accuracy of 100% across tests, underscoring its reliability. The final model was saved using the Joblib library for subsequent deployment.

# Model Deployment
The deployment involved setting up a system that mimics the data recording structure but
operates in real-time. The system captures gesture data, extracts features, and loads the trained model to predict and classify the gesture. The classification results are then displayed to the user, facilitating a real-time interactive interface.

# Testing & Evaluation
The deployed model was tested in a real-time to validate its performance. The evaluation
focused on the importance of various features, using Pandas to demonstrate feature correlation through visualization. Features such as X_mean showed high importance, while others like Magnitude_skew were less significant, providing insights into model improvements and optimization.

# Setup
This project has three main python files:]
*  Recording.py: Application that when run will record your desired gestures and trials and save them appropriately to be used for training
*  Model.py: Preprocesses the data from Data.py and trains a ML model, then dumps the model so it can be used in deployment
*  Testing.py: Uses the model generated and records gestures and attempts to classify them
