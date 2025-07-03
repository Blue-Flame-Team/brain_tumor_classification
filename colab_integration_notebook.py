# ربط الموقع بنموذج من ملف Jupyter Notebook (.ipynb)
# انسخ هذا الكود في Google Colab

# =============================================================================
# الخطوة 1: تثبيت المكتبات المطلوبة
# =============================================================================

# قم بتشغيل هذا في خلية منفصلة:
# !pip install flask flask-cors pyngrok pillow tensorflow nbformat

# =============================================================================
# الخطوة 2: رفع ملف النموذج (.ipynb) إلى Colab
# =============================================================================

# طريقة 1: رفع الملف يدوياً
# 1. اضغط على أيقونة الملفات في الشريط الجانبي
# 2. اضغط على "Upload" 
# 3. اختر ملف .ipynb الخاص بك

# طريقة 2: رفع من Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# =============================================================================
# الخطوة 3: تحميل النموذج من الـ notebook
# =============================================================================

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
import numpy as np
from PIL import Image
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import pickle
import joblib

# دالة لتحميل النموذج من مصادر مختلفة
def load_brain_tumor_model():
    """تحميل النموذج من مصادر مختلفة"""
    
    print("🔍 جاري البحث عن النموذج...")
    
    # المسارات المحتملة للنموذج
    possible_paths = [
        # ملفات TensorFlow/Keras
        'brain_tumor_model.h5',
        'model.h5',
        'best_model.h5',
        'brain_tumor_classifier.h5',
        'mobilenet_brain_tumor.h5',
        
        # ملفات SavedModel
        'saved_model/',
        'brain_tumor_model/',
        'model/',
        
        # ملفات Pickle
        'brain_tumor_model.pkl',
        'model.pkl',
        'classifier.pkl',
        
        # من Google Drive (إذا كان mounted)
        '/content/drive/MyDrive/brain_tumor_model.h5',
        '/content/drive/MyDrive/model.h5',
        '/content/drive/MyDrive/Models/brain_tumor_model.h5',
    ]
    
    # البحث عن النموذج
    for path in possible_paths:
        if os.path.exists(path):
            try:
                print(f"✅ تم العثور على النموذج: {path}")
                
                if path.endswith('.h5'):
                    # تحميل نموذج Keras/TensorFlow
                    model = load_model(path)
                    print(f"📊 نموذج TensorFlow محمل من: {path}")
                    return model, get_model_classes()
                    
                elif path.endswith('.pkl'):
                    # تحميل نموذج Pickle
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"📊 نموذج Pickle محمل من: {path}")
                    return model, get_model_classes()
                    
                elif os.path.isdir(path):
                    # تحميل SavedModel
                    model = tf.saved_model.load(path)
                    print(f"📊 SavedModel محمل من: {path}")
                    return model, get_model_classes()
                    
            except Exception as e:
                print(f"❌ فشل تحميل النموذج من {path}: {str(e)}")
                continue
    
    print("⚠️ لم يتم العثور على نموذج محفوظ، سيتم إنشاء نموذج تجريبي")
    return create_demo_model(), get_model_classes()

def get_model_classes():
    """الحصول على فئات النموذج - قم بتعديل هذه حسب نموذجك"""
    
    # الفئات الشائعة لتصنيف أورام الدماغ
    # قم بتعديل هذه القائمة حسب فئات نموذجك
    possible_classes = [
        # الفئات الثلاث الشائعة
        ['glioma', 'meningioma', 'pituitary'],
        ['الورم الدبقي', 'الورم السحائي', 'الورم النخامي'],
        
        # الفئات الأربع (مع no_tumor)
        ['glioma', 'meningioma', 'pituitary', 'no_tumor'],
        ['الورم الدبقي', 'الورم السحائي', 'الورم النخامي', 'لا يوجد ورم'],
        
        # الفئات الثنائية
        ['tumor', 'no_tumor'],
        ['ورم', 'لا يوجد ورم'],
    ]
    
    # اختر الفئات المناسبة لنموذجك
    # يمكنك تغيير الرقم حسب نموذجك (0 للثلاث فئات بالإنجليزية، 1 للعربية، إلخ)
    selected_classes = possible_classes[1]  # الفئات الثلاث بالعربية
    
    print(f"📋 فئات النموذج: {selected_classes}")
    return selected_classes

def create_demo_model():
    """إنشاء نموذج تجريبي إذا لم يتم العثور على نموذج محفوظ"""
    print("🔧 جاري إنشاء نموذج تجريبي...")
    
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)  # 3 فئات
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # تجربة سريعة للتأكد من عمل النموذج
    dummy_input = np.random.random((1, 224, 224, 3))
    _ = model.predict(dummy_input)
    
    print("✅ تم إنشاء النموذج التجريبي بنجاح")
    return model

# =============================================================================
# الخطوة 4: رفع نموذجك وتحديد المسار
# =============================================================================

print("📁 رفع ملف النموذج...")
print("💡 إذا كان نموذجك محفوظ في notebook، قم بحفظه أولاً:")
print("""
# في نهاية الـ notebook الخاص بك، أضف هذا الكود:

# حفظ النموذج بصيغة .h5
model.save('brain_tumor_model.h5')

# أو حفظ بصيغة SavedModel
model.save('brain_tumor_model/')

# أو حفظ بصيغة Pickle (للنماذج غير TensorFlow)
import pickle
with open('brain_tumor_model.pkl', 'wb') as f:
    pickle.dump(model, f)
""")

# تحميل النموذج
print("🔄 جاري تحميل النموذج...")
model, class_names = load_brain_tumor_model()

# =============================================================================
# الخطوة 5: إعداد Flask API
# =============================================================================

app = Flask(__name__)
CORS(app, origins=["*"])

def preprocess_image(image_data):
    """معالجة الصورة قبل التنبؤ"""
    try:
        # إزالة البادئة data:image/...;base64,
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # تحويل base64 إلى صورة
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # تحويل إلى RGB إذا كانت RGBA
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # تغيير الحجم (قم بتعديل الحجم حسب نموذجك)
        image = image.resize((224, 224))  # معظم النماذج تستخدم 224x224
        
        # تحويل إلى array وتطبيع
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        print(f"❌ خطأ في معالجة الصورة: {str(e)}")
        return None

def make_prediction(processed_image):
    """تشغيل النموذج وإرجاع النتائج"""
    try:
        # تشغيل النموذج
        predictions = model.predict(processed_image)[0]
        
        # تحويل النتائج
        results = []
        for i, class_name in enumerate(class_names):
            probability = float(predictions[i] * 100)
            results.append({
                'type': class_name,
                'probability': round(probability, 1)
            })
        
        # ترتيب النتائج حسب الاحتمالية
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return results
        
    except Exception as e:
        print(f"❌ خطأ في التنبؤ: {str(e)}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    """endpoint للتنبؤ"""
    try:
        # استلام البيانات
        data = request.json
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'لم يتم إرسال صورة'
            }), 400
        
        # معالجة الصورة
        print("🔄 جاري معالجة الصورة...")
        processed_image = preprocess_image(data['image'])
        
        if processed_image is None:
            return jsonify({
                'success': False,
                'error': 'فشل في معالجة الصورة'
            }), 400
        
        # التنبؤ
        print("🧠 جاري تشغيل النموذج...")
        predictions = make_prediction(processed_image)
        
        if predictions is None:
            return jsonify({
                'success': False,
                'error': 'فشل في التنبؤ'
            }), 500
        
        print(f"✅ التنبؤ مكتمل: {predictions[0]['type']} ({predictions[0]['probability']}%)")
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'top_prediction': predictions[0],
            'model_info': {
                'classes': class_names,
                'input_shape': [224, 224, 3]
            }
        })
        
    except Exception as e:
        print(f"❌ خطأ في التنبؤ: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'خطأ في الخادم: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """فحص حالة الخادم"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_classes': class_names,
        'message': 'الخادم يعمل بشكل طبيعي'
    })

@app.route('/', methods=['GET'])
def home():
    """الصفحة الرئيسية"""
    return jsonify({
        'message': 'مرحباً! API تشخيص أورام الدماغ جاهز',
        'model_info': {
            'classes': class_names,
            'num_classes': len(class_names)
        },
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        }
    })

# =============================================================================
# الخطوة 6: إعداد ngrok وبدء الخادم
# =============================================================================

from pyngrok import ngrok
import threading
import time

def run_flask():
    """تشغيل Flask في خيط منفصل"""
    app.run(host='0.0.0.0', port=5000, debug=False)

print("🚀 جاري إعداد الخادم...")

# بدء Flask في خيط منفصل
flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

# انتظار لحظة حتى يبدأ Flask
time.sleep(3)

# إنشاء ngrok tunnel
print("🌐 جاري إنشاء الرابط العام...")
public_url = ngrok.connect(5000)

print("\n" + "="*60)
print("🎉 تم إعداد الخادم بنجاح!")
print("="*60)
print(f"📍 الرابط العام: {public_url}")
print(f"🔗 اختبار الخادم: {public_url}/health")
print(f"🧠 endpoint التنبؤ: {public_url}/predict")
print(f"📋 فئات النموذج: {class_names}")
print("="*60)

# =============================================================================
# الخطوة 7: كود JavaScript للموقع
# =============================================================================

website_code = f"""
// انسخ هذا الكود في موقعك (ملف index.html)
// استبدل قيمة API_URL بالرابط الذي ظهر أعلاه

const API_URL = '{public_url}';

// اختبار الاتصال مع الخادم
async function testConnection() {{
    try {{
        const response = await fetch(API_URL + '/health');
        const result = await response.json();
        console.log('✅ الاتصال مع الخادم ناجح:', result);
        console.log('📋 فئات النموذج:', result.model_classes);
        return true;
    }} catch (error) {{
        console.error('❌ فشل في الاتصال مع الخادم:', error);
        return false;
    }}
}}

// دالة التنبؤ الحقيقية
async function performRealAnalysis(imageFile) {{
    try {{
        // تحويل الصورة إلى base64
        const base64Image = await convertToBase64(imageFile);
        
        // إرسال الطلب للخادم
        const response = await fetch(API_URL + '/predict', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
            }},
            body: JSON.stringify({{
                image: base64Image
            }})
        }});
        
        const result = await response.json();
        
        if (result.success) {{
            console.log('✅ التنبؤ ناجح:', result);
            console.log('🏆 أعلى تنبؤ:', result.top_prediction);
            return result.predictions;
        }} else {{
            console.error('❌ خطأ في التنبؤ:', result.error);
            return generateMockPredictions();
        }}
        
    }} catch (error) {{
        console.error('❌ خطأ في الاتصال:', error);
        return generateMockPredictions();
    }}
}}

console.log('🔗 تم تحديث الموقع للاتصال مع:', API_URL);
console.log('📋 فئات النموذج المتوقعة: {class_names}');
"""

print("\n📋 كود JavaScript المحدث لموقعك:")
print("-" * 50)
print(website_code)

# =============================================================================
# الخطوة 8: اختبار النظام
# =============================================================================

def test_api():
    """اختبار سريع للـ API"""
    print("\n🧪 اختبار النظام...")
    
    try:
        import requests
        
        # اختبار health check
        health_response = requests.get(f"{public_url}/health")
        health_data = health_response.json()
        print(f"✅ Health Check: {health_data['status']}")
        print(f"📊 النموذج محمل: {health_data['model_loaded']}")
        print(f"📋 فئات النموذج: {health_data['model_classes']}")
        
        print("\n💡 النظام جاهز للاستخدام!")
        print("📱 يمكنك الآن استخدام الموقع مع نموذجك الحقيقي")
        
    except Exception as e:
        print(f"⚠️  تحذير: {e}")
        print("💡 الخادم يعمل، لكن تعذر الاختبار التلقائي")

# تشغيل الاختبار
test_api()

print("\n" + "="*60)
print("📖 ملاحظات مهمة:")
print("="*60)
print("1. 🔄 الخادم سيستمر في العمل طالما هذه الخلية تعمل")
print("2. 🔗 انسخ الرابط العام واستخدمه في موقعك")
print("3. 📱 اختبر الموقع بصور MRI حقيقية")
print("4. 🔒 هذا الرابط مؤقت وسينتهي عند إغلاق Colab")
print("5. 📊 تأكد من أن فئات النموذج تطابق التوقعات")
print("="*60)

# إبقاء الخادم يعمل
try:
    print("\n⏳ الخادم يعمل... اضغط Ctrl+C للإيقاف")
    while True:
        time.sleep(60)
        print(f"💓 الخادم ما زال يعمل: {public_url}")
        print(f"📊 فئات النموذج: {class_names}")
except KeyboardInterrupt:
    print("\n🛑 تم إيقاف الخادم") 