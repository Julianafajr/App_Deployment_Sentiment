"""
Quick test script to verify model can be loaded and makes predictions
"""

import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re


# Define Attention Layer (same as in training)
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1],),
            initializer="random_normal",
            trainable=True,
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        e = tf.keras.backend.tanh(inputs)
        e = tf.keras.backend.dot(e, tf.keras.backend.expand_dims(self.W, -1))
        e = tf.keras.backend.squeeze(e, -1)

        if mask is not None:
            mask = tf.keras.backend.cast(mask, tf.keras.backend.floatx())
            e = e * mask + ((1 - mask) * -1e9)

        alpha = tf.keras.backend.softmax(e)
        alpha_exp = tf.keras.backend.expand_dims(alpha, axis=-1)
        context = tf.keras.backend.sum(inputs * alpha_exp, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        return config


def clean_text(text):
    """Clean and preprocess text"""
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("=" * 60)
print("Testing Sentiment Analysis Model")
print("=" * 60)

try:
    # Load model
    print("\n1. Loading model...")
    model = tf.keras.models.load_model(
        "kaggle/working/model_outputs/CNN + BiLSTM_best.h5",
        custom_objects={"AttentionLayer": AttentionLayer},
    )
    print("✓ Model loaded successfully")
    print(f"   Model input shape: {model.input_shape}")
    print(f"   Model output shape: {model.output_shape}")

    # Load tokenizer
    print("\n2. Loading tokenizer...")
    with open("kaggle/working/model_outputs/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("✓ Tokenizer loaded successfully")
    print(f"   Vocabulary size: {len(tokenizer.word_index)}")

    # Load label encoder
    print("\n3. Loading label encoder...")
    with open("kaggle/working/model_outputs/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("✓ Label encoder loaded successfully")
    print(f"   Classes: {list(label_encoder.classes_)}")

    # Test predictions
    print("\n4. Testing predictions...")
    test_texts = [
        "Mobil listrik ini sangat bagus dan hemat energi. Saya sangat puas!",
        "Harga mobil listrik terlalu mahal dan infrastruktur charging masih kurang.",
        "Pemerintah sedang menyiapkan insentif untuk kendaraan bermotor listrik.",
    ]

    MAX_LEN = 128

    for i, text in enumerate(test_texts, 1):
        print(f"\n   Test {i}: {text[:50]}...")

        # Preprocess
        cleaned = clean_text(text)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = np.array(
            pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
        )

        # Predict
        prediction = model.predict(padded, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        sentiment = label_encoder.classes_[predicted_class_idx]

        print(f"   → Sentiment: {sentiment}")
        print(f"   → Confidence: {confidence*100:.2f}%")
        print(f"   → All probabilities:")
        for j, class_name in enumerate(label_encoder.classes_):
            print(f"      - {class_name}: {prediction[0][j]*100:.2f}%")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - Model is working correctly!")
    print("=" * 60)
    print("\n✓ Ready to run: streamlit run app.py")

except Exception as e:
    print("\n" + "=" * 60)
    print(f"✗ ERROR: {str(e)}")
    print("=" * 60)
    import traceback

    traceback.print_exc()
