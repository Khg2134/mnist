import numpy as np
import gzip
import os
import cv2

def load_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = np.frombuffer(f.read(16), dtype=">u4")
        assert magic == 2051, f"bad magic: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)

def load_labels(path):
    with open(path, "rb") as f:
        magic, n = np.frombuffer(f.read(8), dtype=">u4")
        assert magic == 2049, f"bad magic: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8)

x_train = load_images("data/train-images.idx3-ubyte")
y_train = load_labels("data/train-labels.idx1-ubyte")
x_test  = load_images("data/t10k-images.idx3-ubyte")
y_test  = load_labels("data/t10k-labels.idx1-ubyte")
img_test = cv2.imread('data/test_digit_img.png', cv2.IMREAD_GRAYSCALE) 

x_train = x_train.reshape(-1, 28 * 28)
x_test  = x_test.reshape(-1, 28 * 28)
img_test = img_test.reshape(28 * 28)

x_train_norm = x_train.astype(np.float32) / 255.0
x_test_norm = x_test.astype(np.float32) / 255.0
img_test_norm = img_test.astype(np.float32) / 255.0


class net :
    def __init__(self, n_hiddens, n_input_args, n_outputs, lr):
        self.n_hiddens = n_hiddens
        self.n_input_args = n_input_args
        self.n_outputs = n_outputs
        self.lr = lr

        self.w_h = np.random.uniform(-4, 4, (n_hiddens, n_input_args)) * np.sqrt(2 / n_input_args)
        self.w_o = np.random.uniform(-4, 4, (n_outputs, n_hiddens)) * np.sqrt(2 / n_hiddens)
        self.b_h = np.zeros(n_hiddens)
        self.b_o = np.zeros(n_outputs)

        self.x = None
        self.z_h = None
        self.a_h = None
        self.z_o = None

    def reLu(self, z):
        return np.maximum(0, z)

    def softmax(self, x):
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


    def reLu_derivative(self, z):
        return (z > 0).astype(float)

    def forward_hidden(self, x):
        z = self.w_h @ self.x + self.b_h
        a = self.reLu(z)

        self.z_h = z
        self.a_h = a

        return a

    def forward_output(self, x):    
        z = self.w_o @ x + self.b_o

        self.z_o = z

        return z

    def forward_pass(self, x):
        self.x = x

        a_h = self.forward_hidden(x)
        z_o = self.forward_output(a_h)

        return z_o

    def backward(self, y, y_t) :
        dz_o = y - y_t

        dw_o = dz_o.reshape(-1, 1) @ self.a_h.reshape(-1, 1).T
        db_o = dz_o

        

        da_h = self.w_o.T @ dz_o
        dz_h = da_h * self.reLu_derivative(self.z_h)
        
        dw_h = dz_h.reshape(-1, 1) @ self.x.reshape(-1, 1).T
        db_h = dz_h

        return dw_o, db_o, dw_h, db_h

    def update(self, dw_o, db_o, dw_h, db_h):
        self.w_o = self.w_o - (self.lr * dw_o)
        self.b_o = self.b_o - (self.lr * db_o)
        self.w_h = self.w_h - (self.lr * dw_h)
        self.b_h = self.b_h - (self.lr * db_h)

    def save(self):
        np.savez(
            'model.npz',
            w_h = self.w_h,
            w_o = self.w_o, 
            b_h = self.b_h, 
            b_o = self.b_o
        )

    def load(self):
        if os.path.exists('model.npz'):
            data = np.load('model.npz')
            self.w_h = data['w_h']
            self.w_o = data['w_o']
            self.b_h = data['b_h']
            self.b_o = data['b_o']
        else:
            pass

import time
import numpy as np

def train(main_net: net, max_attempts):
    main_net.load()

    total_samples = len(x_train_norm)
    print(f"[TRAIN] Starting training | epochs={max_attempts} | samples/epoch={total_samples} | total_steps={max_attempts * total_samples}")

    train_start = time.time()

    for i in range(max_attempts):
        epoch_start = time.time()
        epoch_loss = 0.0
        correct = 0
        log_interval = max(1, total_samples // 10)  # log ~10 times per epoch

        for j in range(total_samples):
            x = x_train_norm[j]
            y_t_idx = y_train[j]
            y_t = np.zeros(10)
            y_t[y_t_idx] = 1

            z_o = main_net.forward_pass(x)
            y = main_net.softmax(z_o)

            # --- loss / accuracy tracking ---
            eps = 1e-12
            loss = -np.sum(y_t * np.log(y + eps))
            epoch_loss += loss
            pred = int(np.argmax(y))
            if pred == y_t_idx:
                correct += 1

            # --- progress log ---
            if (j + 1) % log_interval == 0 or (j + 1) == total_samples:
                avg_loss = epoch_loss / (j + 1)
                acc = correct / (j + 1)
                elapsed = time.time() - epoch_start
                print(
                    f"[TRAIN] epoch {i+1}/{max_attempts} "
                    f"| step {j+1}/{total_samples} "
                    f"| loss={avg_loss:.4f} "
                    f"| acc={acc:.4f} "
                    f"| pred={pred} target={y_t_idx} "
                    f"| elapsed={elapsed:.2f}s"
                )

            dw_o, db_o, dw_h, db_h = main_net.backward(y, y_t)
            main_net.update(dw_o, db_o, dw_h, db_h)

        # --- end of epoch ---
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / total_samples
        acc = correct / total_samples
        print(
            f"[TRAIN] === epoch {i+1}/{max_attempts} done "
            f"| avg_loss={avg_loss:.4f} "
            f"| acc={acc:.4f} "
            f"| time={epoch_time:.2f}s "
            f"| total_time={time.time() - train_start:.2f}s ==="
        )

        main_net.save()
        print(f"[TRAIN] checkpoint saved after epoch {i+1}")

    total_time = time.time() - train_start
    main_net.save()
    print(f"[TRAIN] Training complete | total_time={total_time:.2f}s | final model saved")

def test(main_net: net):
    main_net.load()
    total_samples = len(x_test_norm)
    print(f"[TEST] Starting | samples={total_samples}")

    total_loss = 0.0
    correct = 0
    log_interval = max(1, total_samples // 10)

    for j in range(total_samples):
        x = x_test_norm[j]
        y_t_idx = y_test[j]
        y_t = np.zeros(10)
        y_t[y_t_idx] = 1

        z_o = main_net.forward_pass(x)
        y = main_net.softmax(z_o)

        predicted_idx = np.argmax(y)

        eps = 1e-12
        loss = -np.sum(y_t * np.log(y + eps))
        total_loss += loss
        pred = int(np.argmax(y))
        if pred == y_t_idx:
            correct += 1

        if (j + 1) % log_interval == 0 or (j + 1) == total_samples:
            print(
                f"[TEST] step {j+1}/{total_samples} "
                f"| loss={loss:.4f} "
                f"| avg_loss={total_loss/(j+1):.4f} "
                f"| acc={correct/(j+1):.4f}"
                f"| answer={y_t_idx}"
                f"| predicted_answer={predicted_idx}"
            )

    print(
        f"[TEST] done | avg_loss={total_loss/total_samples:.4f} "
        f"| acc={correct/total_samples:.4f} ({correct}/{total_samples})"
    )

def test_img(main_net : net):
    x = img_test_norm
    y_t_idx = 1
    y_t = np.zeros(10)
    y_t[y_t_idx] = 1

    z_o = main_net.forward_pass(x)
    y = main_net.softmax(z_o)

    predicted_idx = np.argmax(y)

    eps = 1e-12
    loss = -np.sum(y_t * np.log(y + eps))

    print(
        f"[TEST] "
        f"| loss={loss:.4f} "
        f"| answer={y_t_idx}"
        f"| predicted_answer={predicted_idx}"
    )

def main():
    main_net = net(500, 784, 10, 1e-3)
    # train(main_net, 500)
    # test(main_net)  
    test_img(main_net)

if __name__ == "__main__":
    main()