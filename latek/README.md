## Learning to grasp unknown objects with a robotic arm from experience
###### Luca Monorchio's master thesis, July 2018

#### Abstract
The evolution of technology has opened new revolutionary scenarios in the field of robotics, making it underhand the human presence in favor of the use of increasingly efficient and automated systems. The 
possibility of using robots capable of grasping complicated, as well as heavy, objects pushes the industrial sector towards automatic palletizing, replacing human labour as far as possible. The main reasons are
purely economic, aimed at saving both time and production, which are essential in this type of sector. The robotics grasping research regards to make machines understand which is the best way to grasp objects 
independently, in little terms, the estimation of the best grasping pose. We are therefore faced with a clear problem of computer vision, a fundamental element to allow the robot to identify the region in the 
space where the gripper will have to perform the grasping. Learning systems such as deep learning, especially CNNs, have reached the state of the art in robot grasping. Using the resources made available in the
\emph{RoCoCo} laboratory, \emph{Sapienza}, University of Rome, including a \emph{Microsoft Kinect2} camera rgb-depth and a \emph{ROBOX} 6 d.o.f. antropomorphic robot manipulator, this thesis work aims at learning 
to grasp unknow objects from experience. Hence this project introduces a new dataset consisting of 4000 acquisitions in which each scenario is related to a robot attempting to grasp an object. This dataset provides
both rbg and depth registered images and also the data information about the point of grasping. In addition, this work also introduces a custom Convolutional Neural Network suitable for the features of the dataset
and trained by using TensorFlow deep learning framework. All this immense work aims at converging the predicted outputs by the network on the same wavelength of which the same title of this thesis work makes 
use:  learning to grasp unknow objects with a robotic arm from experience.