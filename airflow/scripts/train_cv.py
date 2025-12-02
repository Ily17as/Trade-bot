def train_cv():
    import time
    import os

    # Здесь твой CV pipeline
    print("Training CV model...")
    time.sleep(3)

    save_path = os.getenv("PROJECT_PATH") + "/models/cv/model_cv.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        f.write("dummy cv model")

    print("CV model saved to:", save_path)
