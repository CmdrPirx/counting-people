# counting-people
Some code, figures, fine-tuned EfficientDet D0 object detection model, and PDF file for bachelor thesis titled Counting People In The Street From Video.
Abstract:
The thesis aims to determine and compare the performance of different crowd counting methods, both shallow and deep on different kinds of crowds. In order to do that, the thesis investigates two approaches, a shallow approach based on predicting human count using foreground pixel counting, and a deep learning approach that utilizes EfficientDet object detection model to detect and count humans. The pixel counting approach shows promising results on PETS2009 crowd dataset achieving mean average error between predicted and real human count of 3.35 on a video of a dense crowd featuring up to 35 people. We also discover that while fine-tuning a pre-trained object detection model can improve its detection ability, human detection based on detecting whole body performs poorly on dense crowds featuring occlusion. 

PETS2009 Dataset available at http://cs.binghamton.edu/~mrldata/pets2009
