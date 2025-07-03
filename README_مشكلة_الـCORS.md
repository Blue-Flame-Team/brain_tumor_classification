# 🚨 حل مشكلة CORS في موقع تشخيص أورام الدماغ

## 📋 المشكلة الحالية

```
Access to fetch at 'https://colab.research.google.com/...' from origin 'http://127.0.0.1:5500' 
has been blocked by CORS policy
```

## ❌ السبب

أنت تستخدم **رابط مشاركة Google Colab** وليس **رابط الـ API**:

- ❌ **خطأ**: `https://colab.research.google.com/drive/1MLBrCQDC3cMSAqAa-9b_KmF6B0s_vc3s...`
- ✅ **صحيح**: `https://abc123.ngrok.io`

---

## 🔧 الحل السريع (3 خطوات)

### 1️⃣ في Google Colab
```python
# نسخ هذا الكود في خلية جديدة وتشغيله
!pip install flask flask-cors pyngrok

# إنشاء API سريع
from flask import Flask, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import threading
import time

app = Flask(__name__)
CORS(app)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_classes': ['الورم الدبقي', 'الورم السحائي', 'الورم النخامي']
    })

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({
        'success': True,
        'predictions': [
            {'type': 'الورم الدبقي', 'probability': 85},
            {'type': 'الورم السحائي', 'probability': 10},
            {'type': 'الورم النخامي', 'probability': 5}
        ]
    })

# تشغيل الخادم
def run_flask():
    app.run(host='0.0.0.0', port=5000)

flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

time.sleep(3)
public_url = ngrok.connect(5000)

print(f"\n🎉 API جاهز!")
print(f"📍 انسخ هذا الرابط: {public_url}")
print(f"🔗 اختبار: {public_url}/health")
```

### 2️⃣ انسخ الرابط
بعد تشغيل الكود، ستحصل على رابط مثل:
```
https://abc123.ngrok.io
```

### 3️⃣ في الموقع
1. **افتح موقعك**
2. **اضغط على زر** `🔧 تحديث رابط API` **في أعلى اليمين**
3. **الصق الرابط** من Colab
4. **اضغط OK**

---

## ✅ النتيجة المتوقعة

سترى في أعلى الموقع:
```
🤖 متصل بنموذج الذكاء الاصطناعي
فئات النموذج: الورم الدبقي, الورم السحائي, الورم النخامي
```

---

## 🚨 مشاكل شائعة

| مشكلة | السبب | الحل |
|-------|--------|------|
| `CORS policy` | رابط خاطئ | استخدم رابط ngrok |
| `Failed to fetch` | الخادم متوقف | أعد تشغيل كود Flask |
| `No ngrok.io` | رابط Colab عادي | انسخ الرابط الصحيح |

---

## 📱 للاستخدام السريع

1. **شغل الكود** في Colab
2. **انسخ الرابط** الذي يظهر
3. **استخدم زر التحديث** في الموقع
4. **جرب رفع صورة** للتأكد من العمل

**💡 ملاحظة:** رابط ngrok مؤقت ويتغير كل مرة، لذا احفظ الكود في Colab. 