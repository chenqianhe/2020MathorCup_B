

import paddlex as pdx

print("Loading model...")
model = pdx.deploy.Predictor('inference_model', use_gpu=True)
print("Model loaded.")


for i in []:
    result = model.predict(i)
    pdx.seg.visualize(i, result, weight=0.0, save_dir='./')
