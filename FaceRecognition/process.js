// Memuat model TensorFlow.js
let model;

async function loadModel() {
    model = await tf.loadLayersModel('Model/model.json');
    console.log('Model Loaded!');
}

// Mengubah gambar menjadi tensor dan melakukan pra-pemrosesan
function processImage(image) {
    const img = tf.browser.fromPixels(image);
    const resized = tf.image.resizeBilinear(img, [112, 112]); // Resize gambar ke 112x112
    const normalized = resized.div(tf.scalar(255.0)); // Normalisasi nilai piksel ke rentang [0, 1]
    const batched = normalized.expandDims(0); // Menambahkan dimensi batch
    return batched;
}

// Menangani prediksi ketika pengguna mengupload gambar
document.getElementById('predictImage').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = async (e) => {
            const imgElement = document.createElement('img');
            imgElement.src = e.target.result;
            imgElement.onload = async () => {
                // Menampilkan gambar yang diupload
                document.getElementById('imageContainer').innerHTML = '';
                document.getElementById('imageContainer').appendChild(imgElement);

                // Proses gambar dan prediksi
                const imageTensor = processImage(imgElement);
                const output = model.predict(imageTensor);

                // Mengambil hasil output dan menampilkannya
                output.array().then(prediction => {
                    const result = prediction[0][0] > prediction[0][1] ? 'Prediksi: Bermasker' : 'Prediksi: Tanpa Masker';
                    document.getElementById('predictionResult').textContent = result;

                    // Menampilkan gambar hasil prediksi
                    const outputImage = tf.browser.toPixels(output.squeeze(), document.createElement('canvas'));
                    document.getElementById('imageContainer').appendChild(outputImage);
                });
            };
        };
        reader.readAsDataURL(file);
    } else {
        alert('Please upload an image file!');
    }
});

// Memuat model ketika halaman dimuat
window.onload = () => {
    loadModel();
};
