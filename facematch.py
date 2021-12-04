import face_recognition

image_of_dipok = face_recognition.load_image_file('./img/known/Dipok.jpg')
dipok_face_encoding = face_recognition.face_encodings(image_of_dipok)[0]

unknown_image = face_recognition.load_image_file(
    './img/unknown/2.jpg')
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces(
    [dipok_face_encoding], unknown_face_encoding)

if results[0]:
    print('This is Dipok')
else:
    print('This is NOT Dipok')
