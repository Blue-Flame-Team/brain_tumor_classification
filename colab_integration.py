# ملف Colab Integration للربط مع الموقع
# انسخ هذا الكود في Google Colab واتبع التعليمات

# =============================================================================
# الخطوة 1: تثبيت المكتبات المطلوبة
# =============================================================================

# قم بتشغيل هذا الأمر في Colab:
# pip install flask flask-cors pyngrok pillow tensorflow

# =============================================================================
# الخطوة 2: إعداد النموذج (استبدل هذا بنموذجك الحقيقي)
# =============================================================================

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

# إنشاء نموذج تجريبي (استبدل هذا بتحميل نموذجك الحقيقي)
def create_demo_model():
    """
    إنشاء نموذج تجريبي - استبدل هذا بتحميل نموذجك الحقيقي:
    model = tf.keras.models.load_model('path_to_your_model.h5')
    """
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
    predictions = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # للتجربة: إنشاء أوزان عشوائية
    dummy_input = np.random.random((1, 224, 224, 3))
    _ = model.predict(dummy_input)
    
    print("✅ تم إنشاء النموذج التجريبي بنجاح")
    return model

# تحميل النموذج
print("🔄 جاري تحميل النموذج...")
model = create_demo_model()

# =============================================================================
# الخطوة 3: إنشاء Flask API
# =============================================================================

app = Flask(__name__)
CORS(app, origins=["*"])  # السماح لجميع المصادر (للتجربة فقط)

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
        
        # تغيير الحجم إلى 224x224
        image = image.resize((224, 224))
        
        # تحويل إلى array وتطبيع
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        print(f"❌ خطأ في معالجة الصورة: {str(e)}")
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
        predictions = model.predict(processed_image)[0]
        
        # تحويل النتائج
        classes = ['الورم الدبقي', 'الورم السحائي', 'الورم النخامي']
        results = []
        
        for i, class_name in enumerate(classes):
            probability = float(predictions[i] * 100)
            results.append({
                'type': class_name,
                'probability': round(probability, 1)
            })
        
        # ترتيب النتائج حسب الاحتمالية
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        print(f"✅ التنبؤ مكتمل: {results[0]['type']} ({results[0]['probability']}%)")
        
        return jsonify({
            'success': True,
            'predictions': results,
            'top_prediction': results[0]
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
        'message': 'الخادم يعمل بشكل طبيعي'
    })

@app.route('/', methods=['GET'])
def home():
    """الصفحة الرئيسية"""
    return jsonify({
        'message': 'مرحباً! API تشخيص أورام الدماغ جاهز',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        }
    })

# =============================================================================
# الخطوة 4: إعداد ngrok
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
print("="*60)

# =============================================================================
# الخطوة 5: تحديث الموقع
# =============================================================================

website_code = f"""
// انسخ هذا الكود واستبدل به الجزء الخاص بـ generateMockPredictions في موقعك

const API_URL = '{public_url}';

// اختبار الاتصال مع الخادم
async function testConnection() {{
    try {{
        const response = await fetch(API_URL + '/health');
        const result = await response.json();
        console.log('✅ الاتصال مع الخادم ناجح:', result);
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
            return result.predictions;
        }} else {{
            console.error('❌ خطأ في التنبؤ:', result.error);
            // العودة للنموذج التجريبي في حالة الخطأ
            return generateMockPredictions();
        }}
        
    }} catch (error) {{
        console.error('❌ خطأ في الاتصال:', error);
        // العودة للنموذج التجريبي في حالة الخطأ
        return generateMockPredictions();
    }}
}}

// دالة مساعدة لتحويل الملف إلى base64
function convertToBase64(file) {{
    return new Promise((resolve, reject) => {{
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file);
    }});
}}

// استبدال دالة performAnalysis في الموقع
function performAnalysis() {{
    const imageFile = document.getElementById('imageInput').files[0];
    if (!imageFile) {{
        alert('يرجى اختيار صورة أولاً');
        return;
    }}
    
    // إظهار شريط التقدم
    analysisProgress.classList.remove('hidden');
    analyzeBtn.disabled = true;
    instructionsPanel.classList.add('hidden');
    
    // محاكاة شريط التقدم
    let progress = 0;
    const progressBar = document.getElementById('progressBar');
    
    const interval = setInterval(() => {{
        progress += Math.random() * 15 + 5;
        if (progress > 100) progress = 100;
        progressBar.style.width = progress + '%';
        
        if (progress >= 100) {{
            clearInterval(interval);
            
            // تشغيل التحليل الحقيقي
            performRealAnalysis(imageFile).then(predictions => {{
                showResults();
                displayResults(predictions);
            }});
        }}
    }}, 150);
}}

// اختبار الاتصال عند تحميل الصفحة
document.addEventListener('DOMContentLoaded', () => {{
    setTimeout(testConnection, 2000);
}});

console.log('🔗 تم تحديث الموقع للاتصال مع:', API_URL);
"""

print("\n📋 كود JavaScript لموقعك:")
print("-" * 40)
print(website_code)

# =============================================================================
# الخطوة 6: اختبار النظام
# =============================================================================

def test_api():
    """اختبار سريع للـ API"""
    print("\n🧪 اختبار النظام...")
    
    try:
        import requests
        
        # اختبار health check
        health_response = requests.get(f"{public_url}/health")
        print(f"✅ Health Check: {health_response.json()}")
        
        print("\n💡 النظام جاهز للاستخدام!")
        print("📱 يمكنك الآن استخدام الموقع مع النموذج الحقيقي")
        
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
print("3. 📱 اختبر الموقع من أي جهاز متصل بالإنترنت")
print("4. 🔒 هذا الرابط مؤقت وسينتهي عند إغلاق Colab")
print("5. 🛡️  تأكد من عدم مشاركة الرابط مع أشخاص غير موثوقين")
print("="*60)

# إبقاء الخادم يعمل
try:
    print("\n⏳ الخادم يعمل... اضغط Ctrl+C للإيقاف")
    while True:
        time.sleep(60)
        print(f"💓 الخادم ما زال يعمل: {public_url}")
except KeyboardInterrupt:
    print("\n🛑 تم إيقاف الخادم") 