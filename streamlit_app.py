import io
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Классификатор зданий Гомеля",
    page_icon="🏛",
    layout="centered",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_REPO = "ILIAILIAII/zdanija-weights"
WEIGHTS_NAME = "zdanija.weights.h5"
IMG_SIZE = 224

CLASS_NAMES = [
    "Башня обозрения",
    "Дворец Румянцевых — Паскевичей",
    "Зимний сад",
    "Петропавловский собор",
    "Часовня-усыпальница князей Паскевичей",
]

CLASS_INFO = [
    (
        "Историческая смотровая башня высотой 40 метров в парке Гомельского "
        "дворцово-паркового ансамбля. Построена в XIX веке как дымовая труба "
        "сахарного завода князя Паскевича, позже превращена в парковую башню. "
        "Наверх ведёт винтовая лестница из 204 ступеней, сверху — открытая "
        "круговая площадка с панорамным видом на реку Сож и город. "
        "Билет стоит около 7 BYN."
    ),
    (
        "Главная достопримечательность Гомеля и композиционный центр "
        "дворцово-паркового ансамбля, памятник архитектуры классицизма "
        "конца XVIII–XIX веков. Строительство начато в 1777 году по приказу "
        "фельдмаршала Петра Румянцева, в XIX веке имение выкупил Иван Паскевич "
        "и масштабно реконструировал здание. Сегодня в залах работает музей "
        "с историческими интерьерами и коллекциями; здание изображено на "
        "банкноте номиналом 20 рублей. Билет — около 10 BYN."
    ),
    (
        "Историческая оранжерея XIX века в южной части ансамбля рядом "
        "с Лебяжьим прудом. В 1877 году один из цехов сахарного завода "
        "князя Паскевича переоборудовали под оранжерею. Внутри стены выложены "
        "природными минералами, под стеклянной крышей собрана коллекция из "
        "18 видов субтропических растений (магнолии, лимоны, кофейные деревья, "
        "инжир); самая старая пальма хамеропс посажена в 1888 году. "
        "Билет — около 7 BYN."
    ),
    (
        "Кафедральный собор Гомельской епархии, выдающийся памятник "
        "классицизма начала XIX века. Заложен в 1809 году по инициативе графа "
        "Николая Румянцева по проекту архитектора Джона Кларка, освящён "
        "в 1819 году. Здание имеет форму креста с доминирующим куполом "
        "и фасадами с шестиколонными портиками дорического ордера. В советское "
        "время использовалось как планетарий, в 1989 году возвращено верующим. "
        "Главные святыни — мощи преподобной Манефы Гомельской. Вход свободный."
    ),
    (
        "Уникальный памятник псевдорусского стиля конца XIX века в северной "
        "части ансамбля рядом с Петропавловским собором. Комплекс возведён "
        "в 1870–1889 годах по проекту архитектора Максимилиана Месмахера "
        "для захоронения членов княжеской семьи Паскевичей. Состоит из наземной "
        "часовни, похожей на сказочный терем с майоликовыми плитками и мозаикой, "
        "и подземного склепа, где покоятся восемь представителей рода, включая "
        "фельдмаршала Ивана Паскевича. Сегодня открыт как музейный павильон, "
        "билет — около 5 BYN."
    ),
]

MOBILE_CSS = """
<style>
.block-container {max-width: 760px; padding-top: 1rem; padding-bottom: 1rem;}
h1 {font-size: 1.7rem !important;}
h1, h2, h3 {font-family: inherit;}
[data-testid="stButton"] button {min-height: 46px; width: 100%; font-size: 1rem;}
[data-testid="stFileUploaderDropzone"] {min-height: 90px;}
[data-testid="stProgress"] > div > div > div > div {height: 14px;}
@media (max-width: 640px) {
  .block-container {padding-left: 0.6rem; padding-right: 0.6rem;}
  h1 {font-size: 1.35rem !important;}
}
</style>
"""


@st.cache_resource
def get_weights_path() -> str:
    for candidate in (
        os.path.join(BASE_DIR, WEIGHTS_NAME),
        os.path.join(BASE_DIR, "..", WEIGHTS_NAME),
    ):
        if os.path.exists(candidate):
            return candidate
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=WEIGHTS_REPO, filename=WEIGHTS_NAME)


@st.cache_resource
def load_model():
    base = tf.keras.applications.VGG19(
        weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    for layer in base.layers:
        layer.trainable = False
    x = tf.keras.layers.Flatten()(base.output)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=predictions)
    model.load_weights(get_weights_path())
    return model


@st.cache_data
def predict(image_bytes: bytes):
    model = load_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    preds = model.predict(arr[np.newaxis, ...], verbose=0)[0]
    return [float(p) for p in preds]


st.markdown(MOBILE_CSS, unsafe_allow_html=True)
st.title("🏛 Классификатор зданий Гомеля")
st.caption(
    "Загрузите фото одного из пяти зданий — нейросеть VGG19 определит класс "
    "и расскажет о здании."
)

tab_upload, tab_camera = st.tabs(["📁 Загрузить фото", "📷 Камера"])
uploaded = tab_upload.file_uploader(
    "Фото здания", type=["jpg", "jpeg", "png", "bmp", "webp"]
)
camera = tab_camera.camera_input("Сделать фото")

img_bytes = None
if uploaded is not None:
    img_bytes = uploaded.getvalue()
elif camera is not None:
    img_bytes = camera.getvalue()

if img_bytes is None:
    st.info("Здесь появится результат после загрузки фото.")
else:
    st.image(
        Image.open(io.BytesIO(img_bytes)).convert("RGB"), width="stretch"
    )

    with st.spinner("Распознаю..."):
        try:
            probs = predict(img_bytes)
        except Exception as e:
            st.error("Ошибка обработки изображения: " + str(e))
            probs = None

    if probs is not None:
        idx = int(np.argmax(probs))

        st.markdown(f"## {CLASS_NAMES[idx]}")

        with st.expander("Краткая информация", expanded=True):
            st.markdown(CLASS_INFO[idx])
