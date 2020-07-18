# Face-recognition

Orginal Image : 

![](https://github.com/harish-2306/Face-recognition/blob/master/Input.jpeg)

Predicted Image:

![alt text](http://url/to/img.png)


Pipleline : 

1. The face is first extracted from the Image. (find_face.py)
2. Facial Kepypoints are extracted and the face is aligned. (landmarks.py)
3. 128 Features are extracted from the face. (feature_extraction.py)
4. Eucledian distance is calculated across all known features and the best which passes the threshold is chosen. (feature_extraction.py)

Useage :
1. python main.py [filename] -> Image with prediction
2. python main.py [filename] -n [name] -> New person added to features.json
3. python main.py [filename] -u [name] -> More features are added to the person
