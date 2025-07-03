from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)
CORS(app)

# تحميل النموذج
MODEL_PATH = 'brain_tumor_model_final.h5'  # تأكد من وضع المسار الصحيح للملف
model = load_model(MODEL_PATH)

# تعريف الفئات
CLASSES = ['لا يوجد ورم', 'يوجد ورم']  # قم بتعديل الفئات حسب نموذجك

def preprocess_image(image_path):
    # قراءة وتجهيز الصورة - قم بتعديل الأبعاد حسب متطلبات نموذجك
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # تطبيع القيم
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # التحقق من وجود ملف في الطلب
        if 'image' not in request.files:
            return jsonify({'error': 'لم يتم تحميل صورة'}), 400
        
        file = request.files['image']
        
        # حفظ الصورة مؤقتاً
        temp_path = 'temp_image.jpg'
        file.save(temp_path)
        
        # تجهيز الصورة
        processed_image = preprocess_image(temp_path)
        
        # التنبؤ
        prediction = model.predict(processed_image)
        predicted_class = CLASSES[int(np.round(prediction[0][0]))]
        confidence = float(prediction[0][0])
        
        # حذف الملف المؤقت
        os.remove(temp_path)
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 