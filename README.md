# تصنيف أورام الدماغ باستخدام الذكاء الاصطناعي
## Brain Tumor Classification using AI

نظام ذكي لتصنيف أورام الدماغ باستخدام تقنيات التعلم العميق والرؤية الحاسوبية.

### المميزات
- تشخيص سريع ودقيق لأورام الدماغ
- واجهة مستخدم سهلة الاستخدام
- دعم كامل للغة العربية
- تقارير تفصيلية للنتائج

### المتطلبات التقنية
- Python 3.8+
- TensorFlow 2.x
- Flask
- HTML5/CSS3
- JavaScript

### كيفية التشغيل
1. تثبيت المتطلبات:
```bash
pip install -r requirements.txt
```

2. تشغيل الخادم:
```bash
python model_server.py
```

3. فتح الموقع في المتصفح:
```
http://localhost:5000
```

### الفريق
- Blue Flame Team

### الترخيص
MIT License

### للتواصل
- GitHub Issues
- Email: [إضافة البريد الإلكتروني]

## الوصف
موقع تقديمي تفاعلي يعرض نموذج الذكاء الاصطناعي لتشخيص أورام الدماغ باستخدام صور الرنين المغناطيسي. يتضمن الموقع عرضاً تقديمياً شاملاً بالإضافة إلى أداة تفاعلية لتجربة تشخيص الأورام.

## المميزات
- ✅ عرض تقديمي تفاعلي مع 12 سلايد
- ✅ تصميم متجاوب يعمل على جميع الأجهزة  
- ✅ رسوم بيانية تفاعلية باستخدام Chart.js
- ✅ أداة رفع وتحليل صور الأورام
- ✅ واجهة عربية جميلة ومهنية
- ✅ تأثيرات بصرية متقدمة

## كيفية التشغيل

### التشغيل المحلي
```bash
# افتح Terminal في مجلد المشروع
cd /path/to/project

# شغل خادم محلي
python -m http.server 8080

# افتح المتصفح على
http://localhost:8080
```

### التشغيل مع Live Server (VS Code)
1. ثبت Live Server extension
2. انقر بزر الماوس الأيمن على index.html
3. اختر "Open with Live Server"

## دمج نموذج Google Colab الحقيقي

### الطريقة الأولى: استخدام ngrok (الأسرع)

#### في Google Colab:
```python
# 1. تدريب نموذجك في Colab
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from PIL import Image
import base64
import io

# 2. إنشاء Flask API
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# تحميل النموذج المدرب
model = tf.keras.models.load_model('your_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استلام الصورة من الموقع
        data = request.json
        image_data = data['image'].split(',')[1]  # إزالة data:image/jpeg;base64,
        
        # تحويل base64 إلى صورة
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # التنبؤ
        predictions = model.predict(image)[0]
        
        # تحويل إلى نسب مئوية
        classes = ['الورم الدبقي', 'الورم السحائي', 'الورم النخامي']
        results = []
        for i, class_name in enumerate(classes):
            results.append({
                'type': class_name,
                'probability': float(predictions[i] * 100)
            })
        
        # ترتيب النتائج حسب الاحتمالية
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'success': True,
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# 3. تشغيل الخادم
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### إعداد ngrok:
```python
# في خلية جديدة في Colab
!pip install pyngrok
from pyngrok import ngrok

# إنشاء tunnel
public_url = ngrok.connect(5000)
print(f"API متاح على: {public_url}")
```

#### تعديل الموقع:
```javascript
// في ملف index.html، استبدل دالة generateMockPredictions:
async function performRealAnalysis(imageFile) {
    const formData = new FormData();
    
    // تحويل الصورة إلى base64
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 224;
    canvas.height = 224;
    
    const img = new Image();
    img.onload = async () => {
        ctx.drawImage(img, 0, 0, 224, 224);
        const base64Image = canvas.toDataURL('image/jpeg');
        
        try {
            const response = await fetch('YOUR_NGROK_URL/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: base64Image
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                displayResults(result.predictions);
            } else {
                console.error('خطأ في التحليل:', result.error);
                // العودة للنموذج التجريبي
                displayResults(generateMockPredictions());
            }
        } catch (error) {
            console.error('خطأ في الاتصال:', error);
            // العودة للنموذج التجريبي
            displayResults(generateMockPredictions());
        }
    };
    
    img.src = URL.createObjectURL(imageFile);
}
```

### الطريقة الثانية: استخدام Hugging Face Spaces

#### 1. إنشاء Space في Hugging Face:
```python
# app.py
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# تحميل النموذج
model = tf.keras.models.load_model('your_model.h5')

def predict_tumor(image):
    # معالجة الصورة
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # التنبؤ
    predictions = model.predict(image)[0]
    
    # تحويل إلى نتائج
    classes = ['الورم الدبقي', 'الورم السحائي', 'الورم النخامي']
    results = {classes[i]: float(predictions[i] * 100) for i in range(3)}
    
    return results

# إنشاء واجهة Gradio
iface = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="تشخيص أورام الدماغ"
)

if __name__ == "__main__":
    iface.launch()
```

#### 2. دمج مع الموقع:
```javascript
// استخدام Hugging Face Inference API
async function predictWithHuggingFace(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    const response = await fetch('https://YOUR_USERNAME-brain-tumor-classifier.hf.space/api/predict', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    return result;
}
```

### الطريقة الثالثة: استخدام TensorFlow.js (يعمل في المتصفح)

#### 1. تحويل النموذج:
```python
# في Colab
import tensorflowjs as tfjs

# تحويل النموذج إلى TensorFlow.js
tfjs.converters.save_keras_model(model, 'model_tfjs')

# تحميل الملفات وحفظها في مجلد الموقع
```

#### 2. استخدام النموذج في الموقع:
```javascript
// إضافة TensorFlow.js
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>

// تحميل النموذج
let model;
async function loadModel() {
    model = await tf.loadLayersModel('./model_tfjs/model.json');
    console.log('تم تحميل النموذج بنجاح');
}

// التنبؤ باستخدام TensorFlow.js
async function predictWithTensorFlowJS(imageElement) {
    if (!model) {
        console.error('النموذج لم يتم تحميله بعد');
        return generateMockPredictions();
    }
    
    // تحضير الصورة
    const tensor = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims();
    
    // التنبؤ
    const predictions = await model.predict(tensor).data();
    
    // تحويل النتائج
    const classes = ['الورم الدبقي', 'الورم السحائي', 'الورم النخامي'];
    const results = Array.from(predictions).map((prob, index) => ({
        type: classes[index],
        probability: Math.round(prob * 100)
    })).sort((a, b) => b.probability - a.probability);
    
    // تنظيف الذاكرة
    tensor.dispose();
    
    return results;
}

// تحميل النموذج عند بدء الصفحة
document.addEventListener('DOMContentLoaded', () => {
    loadModel();
    // باقي الكود...
});
```

## متطلبات النموذج

### بيانات التدريب:
- صور رنين مغناطيسي للدماغ بحجم 224x224 بكسل
- ثلاث فئات: الورم الدبقي، الورم السحائي، الورم النخامي
- ما لا يقل عن 1000 صورة لكل فئة للحصول على أفضل النتائج

### بنية النموذج المقترحة:
```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# النموذج الأساسي
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# تجميد الطبقات الأساسية
base_model.trainable = False

# إضافة طبقات التصنيف
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# تكوين التدريب
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

## الأمان والاعتبارات

⚠️ **تحذير مهم:**
- هذا النموذج للأغراض التعليمية والبحثية فقط
- لا يجب الاعتماد عليه للتشخيص الطبي الفعلي
- يجب دائماً استشارة طبيب مختص

### أمان البيانات:
- تأكد من عدم حفظ الصور المرفوعة على الخادم
- استخدم HTTPS للاتصالات الآمنة
- اتبع قوانين حماية البيانات الطبية (HIPAA)

## الدعم والمساعدة

للحصول على المساعدة:
- ✉️ البريد الإلكتروني: mnoomidoostar6@gmail.com
- 📱 الهاتف: 01026843062

## الترخيص

هذا المشروع للأغراض التعليمية والبحثية. 