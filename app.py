
import os
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Reshape,
    Multiply,
    Activation,
    Add,
    Concatenate,
    Conv2D,
    Layer,
)
from tensorflow.keras.applications import EfficientNetV2S
from werkzeug.utils import secure_filename

# =========================================================
# CONFIG
# =========================================================
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "models", "efficientnetv2_clean.keras")
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
IMG_SIZE = (200, 200)
THRESHOLD = 0.5
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# =========================================================
# CUSTOM LAYERS
# These replace notebook Lambda layers so model loading is safer.
# =========================================================
class ChannelMean(Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)

    def get_config(self):
        return super().get_config()


class ChannelMax(Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)

    def get_config(self):
        return super().get_config()


# =========================================================
# MODEL BUILDING
# Based on your notebook: EfficientNetV2S + CBAM + binary output
# =========================================================
def channel_attention(input_tensor, ratio=8):
    channel = input_tensor.shape[-1]

    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, channel))(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = Reshape((1, 1, channel))(max_pool)

    shared_dense1 = Dense(channel // ratio, activation="relu")
    shared_dense2 = Dense(channel, activation="linear")

    avg_out = shared_dense2(shared_dense1(avg_pool))
    max_out = shared_dense2(shared_dense1(max_pool))

    attention = Activation("sigmoid")(Add()([avg_out, max_out]))
    return Multiply()([input_tensor, attention])


def spatial_attention(input_tensor):
    avg_pool = ChannelMean()(input_tensor)
    max_pool = ChannelMax()(input_tensor)

    concat = Concatenate()([avg_pool, max_pool])
    attention = Conv2D(1, (7, 7), padding="same", activation="sigmoid")(concat)

    return Multiply()([input_tensor, attention])


def cbam_block(input_tensor, ratio=8):
    x = channel_attention(input_tensor, ratio)
    x = spatial_attention(x)
    return x


def build_efficientnet_v2(use_attention=True):
    base = EfficientNetV2S(
        weights="imagenet",
        include_top=False,
        input_shape=(200, 200, 3),
        include_preprocessing=False,
    )
    base.trainable = False

    x = base.output

    if use_attention:
        x = cbam_block(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    output = Dense(1, activation="sigmoid", dtype="float32")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def load_prediction_model():
    """
    Try to load the saved full model first.
    If that fails because of Lambda serialization issues, rebuild the same
    architecture without Lambda layers and then load weights if possible.
    """
    custom_objects = {
        "ChannelMean": ChannelMean,
        "ChannelMax": ChannelMax,
    }

    # Attempt 1: direct full-model load
    try:
        tf.keras.config.enable_unsafe_deserialization()
    except Exception:
        pass

    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )
    except Exception as e:
        print("Direct model load failed:", e)

    # Attempt 2: rebuild architecture and try loading weights
    try:
        model = build_efficientnet_v2(use_attention=True)
        model.load_weights(MODEL_PATH)
        print("Loaded model weights successfully using rebuilt architecture.")
        return model
    except Exception as e:
        print("Weight loading also failed:", e)
        raise RuntimeError(
            "Could not load model. "
            "If your .keras file was saved with Lambda layers, re-save the model "
            "after replacing Lambda with custom layers or save weights separately."
        ) from e


model = load_prediction_model()


# =========================================================
# HELPERS
# =========================================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(image_path):
    img_array = preprocess_image(image_path)
    probability = float(model.predict(img_array, verbose=0)[0][0])

    if probability > THRESHOLD:
        label = "Leukemia (ALL)"
        confidence = probability * 100
    else:
        label = "Healthy (HEM)"
        confidence = (1 - probability) * 100

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "raw_probability": round(probability, 4),
        "threshold": THRESHOLD,
    }


# =========================================================
# ROUTES
# =========================================================
HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Leukemia Classification</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 700px;
            margin: 40px auto;
            padding: 20px;
            background: #f4f8fb;
        }
        .box {
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        h1 { color: #1f4e79; }
        button {
            background: #1f4e79;
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 8px;
            cursor: pointer;
        }
        input[type=file] {
            margin: 15px 0;
        }
        .note {
            color: #555;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="box">
        <h1>Leukemia Classification using AI</h1>
        <p>Upload a blood smear image to predict whether it is <b>Leukemia (ALL)</b> or <b>Healthy (HEM)</b>.</p>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".png,.jpg,.jpeg,.bmp" required>
            <br>
            <button type="submit">Predict</button>
        </form>
        <p class="note">Image size used by the model: 200 × 200. Preprocessing: pixel values scaled by 1/255.</p>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HOME_HTML)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use png, jpg, jpeg, or bmp."}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(file_path)

    try:
        result = predict_image(file_path)

        # Return JSON for API clients
        if request.headers.get("Accept") == "application/json" or request.is_json:
            return jsonify(result)

        # Return simple HTML for browser form submit
        return render_template_string(
            """
            <html>
            <head>
                <title>Prediction Result</title>
                <style>
                    body { font-family: Arial; max-width: 700px; margin: 40px auto; background: #f4f8fb; padding: 20px; }
                    .card { background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
                    a { text-decoration: none; color: #1f4e79; }
                </style>
            </head>
            <body>
                <div class="card">
                    <h2>Prediction Result</h2>
                    <p><b>Prediction:</b> {{ label }}</p>
                    <p><b>Confidence:</b> {{ confidence }}%</p>
                    <p><b>Raw Probability:</b> {{ raw_probability }}</p>
                    <p><b>Threshold:</b> {{ threshold }}</p>
                    <br>
                    <a href="/">← Predict another image</a>
                </div>
            </body>
            </html>
            """,
            **result,
        )
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    app.run(debug=True)
