# 4dchess
CV 3d illusion chess game made with mediapipe, opencv, and num.py. Matrix style and dynamic chess planes. 

# features 
<li>
  Head Tracking: Uses your webcam and mediapipe to track head movements to create a 3d illusion
</li>
<li>
  Dynamic Depth: Elements warp to add to the 3d effects 
</li>
<li>
  Chess Logic: Allows for sliding moves to be made across boards, adding a challenge to the traditional game
</li>

# requirements 
<li>
  Python 3.7+
</li>
<li>
  Webcam
</li>

# installation 
1. clone the repo with git clone https://github.com/sortiz-kalina/4dchess.git
2. install the required packages with pip install pygame opencv-python mediapipe
3. download face_landmarker task and hand_landmarker task from Mediapipe's model and place it in the root directory

# usage 
1. run the main script
2. move your nose to move the planes
3. move your middle finger to pick a piece, pinch to select
4. move your middle finger to select a spot. pinch to put down
5. all gestures and sensitivity can be adjusted following commented blocks in the main 

