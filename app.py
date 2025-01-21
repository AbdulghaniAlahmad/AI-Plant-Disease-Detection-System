import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# إنشاء تطبيق Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# تحميل الموديلات
yolo_model = YOLO('models/best.pt')
cnn_model = load_model('models/best_model.keras')

# فئات CNN
class_labels = {
    0: 'Potato___Early_blight', 
    1: 'Potato___Late_blight', 
    2: 'Potato___healthy', 
    3: 'Tomato___Early_blight', 
    4: 'Tomato___Late_blight', 
    5: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    6: 'Tomato___Tomato_mosaic_virus',
    7: 'Tomato___healthy'
}

# وصف المرض وطرق الوقاية والعلاج
disease_info = {
    'Potato___Early_blight': {
        'description': 'Erken leke hastalığı, patateslerde yaygın bir mantar hastalığıdır. Yapraklarda koyu kahverengi lekeler oluşur.',
        'prevention': 'Tuzlu suyla sulama yapılabilir, bulaşan bitkiler uzaklaştırılmalıdır.',
        'treatment': 'Fungisit kullanımı önerilir, hasta bitkiler temizlenmeli ve toprağa zarar verilmemelidir.'
    },
    'Potato___Late_blight': {
        'description': 'Geç leke hastalığı, patateslerde solgunluk ve lekelenmeye neden olan bir mantar enfeksiyonudur.',
        'prevention': 'Tuzlu su ile toprak dezenfekte edilmelidir, bitkiler için iyi havalandırma sağlanmalıdır.',
        'treatment': 'Fungisit tedavisi önerilir, etkilenen yapraklar ve bitkiler uzaklaştırılmalıdır.'
    },
    'Potato___healthy': {
        'description': 'Sağlıklı patates bitkisi herhangi bir hastalık belirtisi göstermeyen bitkilerdir.',
        'prevention': 'Sağlıklı bitkiler için düzenli sulama ve uygun bakım gereklidir.',
        'treatment': 'Sağlıklı patates bitkileri herhangi bir tedavi gerektirmez.'
    },
    'Tomato___Early_blight': {
        'description': 'Erken leke hastalığı, domateslerde solgunluk, kahverengi lekeler ve çürümeye neden olan bir mantar hastalığıdır.',
        'prevention': 'Bitkiler arasındaki mesafe arttırılmalı, suyun bitkilerin üstüne düşmesi engellenmelidir.',
        'treatment': 'Fungisit tedavisi önerilir, etkilenen yapraklar temizlenmelidir.'
    },
    'Tomato___Late_blight': {
        'description': 'Geç leke hastalığı, domateslerde lekelenmeye ve yaprakların solmasına neden olan bir enfeksiyon türüdür.',
        'prevention': 'İyi havalandırma sağlanmalı, suyun bitkilerin üstüne dökülmesi engellenmelidir.',
        'treatment': 'Fungisit tedavisi kullanarak hastalık tedavi edilebilir, etkilenen yapraklar temizlenmelidir.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Domates sarı yaprak kıvrılma virüsü (TYLCV), domates bitkilerinde sararma ve yaprak kıvrılmasına neden olan bir virüs hastalığıdır.',
        'prevention': 'Virüs taşıyan böceklerden korunmak için böcek ilacı kullanılmalıdır.',
        'treatment': 'Virüs için tedavi yoktur, etkilenen bitkiler yok edilmelidir.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Domates mozaik virüsü, domateslerde yapraklarda sararma, mozaik desenler ve gelişim bozukluğuna yol açan bir virüs hastalığıdır.',
        'prevention': 'Virüs taşıyan böcekler ve bitkilerden korunmak için böcek ilaçları kullanılmalıdır.',
        'treatment': 'Virüs için tedavi yoktur, etkilenen bitkiler temizlenmeli ve yok edilmelidir.'
    },
    'Tomato___healthy': {
        'description': 'Sağlıklı domates bitkisi, herhangi bir hastalık belirtisi göstermeyen bitkilerdir.',
        'prevention': 'Sağlıklı bitkiler için düzenli sulama ve bakım gereklidir.',
        'treatment': 'Sağlıklı domates bitkileri herhangi bir tedavi gerektirmez.'
    }
}

# دالة التحقق من نوع الملف
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# معالجة الصور لـ CNN
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# الصفحة الرئيسية
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            # حفظ الصورة
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # YOLO تنبؤ
            yolo_results = yolo_model.predict(filepath, conf=0.5)
            detected_classes = [yolo_model.names[int(box.cls[0])] for box in yolo_results[0].boxes]

            if 'Tomato' in detected_classes or 'Potato' in detected_classes:
                # CNN تنبؤ
                image_array = preprocess_image(filepath)
                predictions = cnn_model.predict(image_array)
                predicted_class_index = np.argmax(predictions)
                predicted_class = class_labels[predicted_class_index]
            else:
                predicted_class = "bilinmeyen"

            return render_template('result.html', image_url=url_for('uploaded_file', filename=filename),
                                   detected_class=detected_classes[0] if detected_classes else "Unknown",
                                   prediction=predicted_class, disease_info=disease_info)

    return render_template('index.html')

# عرض الصور المرفوعة
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# تشغيل التطبيق
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)