# كيفية استخراج النموذج من ملف Jupyter Notebook (.ipynb)

## 🎯 خطوات سريعة

### الخطوة 1: فتح ملف النموذج الخاص بك

1. **ارفع ملف** `.ipynb` **إلى Google Colab**
   - اذهب إلى [colab.research.google.com](https://colab.research.google.com)
   - اضغط "File" → "Upload notebook"
   - اختر ملف `.ipynb` الخاص بك

2. **شغل جميع الخلايا** في النموذج
   - اضغط "Runtime" → "Run all"
   - انتظر حتى يتم تدريب/تحميل النموذج

### الخطوة 2: حفظ النموذج

في نهاية notebook الخاص بك، أضف خلية جديدة وشغل هذا الكود:

```python
# حفظ النموذج بصيغة .h5 (الأفضل)
model.save('brain_tumor_model.h5')
print("✅ تم حفظ النموذج بنجاح: brain_tumor_model.h5")

# اختياري: حفظ بصيغ أخرى
# model.save('brain_tumor_model/')  # SavedModel format
# 
# import pickle
# with open('brain_tumor_model.pkl', 'wb') as f:
#     pickle.dump(model, f)
```

### الخطوة 3: تحميل الكود الجديد

1. **انسخ محتوى ملف** `colab_integration_notebook.py` **بالكامل**
2. **ألصقه في خلية جديدة** في Colab
3. **شغل الخلية**

### الخطوة 4: تحديث معلومات النموذج

قبل تشغيل الكود، تأكد من تحديث هذه المعلومات:

```python
# في دالة get_model_classes(), حدث فئات نموذجك:
def get_model_classes():
    possible_classes = [
        ['glioma', 'meningioma', 'pituitary'],           # 0 - إنجليزي
        ['الورم الدبقي', 'الورم السحائي', 'الورم النخامي'],   # 1 - عربي  
        ['glioma', 'meningioma', 'pituitary', 'no_tumor'], # 2 - 4 فئات
        ['tumor', 'no_tumor'],                            # 3 - ثنائي
    ]
    
    # غير الرقم حسب نموذجك:
    selected_classes = possible_classes[1]  # اختر الرقم المناسب
    return selected_classes
```

## 🔧 إعدادات متقدمة

### إذا كان نموذجك يستخدم حجم صورة مختلف:

```python
# في دالة preprocess_image(), غير الحجم:
image = image.resize((224, 224))  # غير إلى حجم نموذجك
```

### إذا كان نموذجك يحتاج معالجة خاصة:

```python
def preprocess_image(image_data):
    # ... الكود الموجود ...
    
    # أضف معالجة خاصة هنا:
    # مثال: تطبيع مختلف
    # image_array = (image_array - 0.5) * 2  # تطبيع [-1, 1]
    
    return image_array
```

## 🚨 حل المشاكل الشائعة

### المشكلة: "لم يتم العثور على النموذج"
**الحل:**
```python
# تأكد من وجود الملف:
import os
print("الملفات الموجودة:", os.listdir('.'))

# إذا كان النموذج في مجلد آخر:
model = load_model('/content/drive/MyDrive/model.h5')
```

### المشكلة: "خطأ في شكل البيانات"
**الحل:**
```python
# تحقق من شكل البيانات المتوقع:
print("شكل البيانات المتوقع:", model.input_shape)

# عدل حجم الصورة وفقاً لذلك
```

### المشكلة: "فئات النموذج غير صحيحة"
**الحل:**
```python
# في نهاية تدريب نموذجك، احفظ الفئات:
class_names = ['فئة1', 'فئة2', 'فئة3']  # فئات نموذجك
import json
with open('class_names.json', 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False)

# ثم في كود الـ API:
with open('class_names.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)
```

## 📋 مثال كامل لنموذج نموذجي

```python
# مثال كامل لحفظ نموذج في نهاية notebook:

# 1. حفظ النموذج
model.save('brain_tumor_model.h5')

# 2. حفظ فئات النموذج
class_names = ['الورم الدبقي', 'الورم السحائي', 'الورم النخامي']
import json
with open('class_names.json', 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False)

# 3. حفظ معلومات النموذج
model_info = {
    'input_shape': list(model.input_shape[1:]),  # [224, 224, 3]
    'num_classes': len(class_names),
    'preprocessing': 'normalize_0_1'  # أو أي معالجة أخرى
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f)

print("✅ تم حفظ جميع الملفات بنجاح!")
```

## 🎯 نصائح للنجاح

1. **تأكد من تشغيل جميع خلايا النموذج** قبل الحفظ
2. **احفظ النموذج في نفس session** التي تم تدريبه فيها
3. **تحقق من أن النموذج محفوظ بنجاح** قبل المتابعة
4. **اختبر النموذج** بصورة تجريبية قبل ربطه بالموقع

---

**💡 مثال سريع:**
إذا كان نموذجك يصنف 4 فئات: `['glioma', 'meningioma', 'pituitary', 'normal']`

غير السطر في `get_model_classes()`:
```python
selected_classes = ['الورم الدبقي', 'الورم السحائي', 'الورم النخامي', 'طبيعي']
```

**🆘 تحتاج مساعدة؟** شارك معي:
1. عدد الفئات في نموذجك
2. أسماء الفئات
3. حجم الصورة المستخدم
4. أي رسائل خطأ تظهر لك 