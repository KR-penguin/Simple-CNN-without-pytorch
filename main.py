import time
import numpy as np
import activation_function as af
import my_cnn_module as cm

x_train, x_test, t_train, t_test = cm.get_test_data()
# 데이터 행렬 크기 = (60000, 784) = (훈련 데이터의 수, 이미지의 픽셀 수)
# 각 데이터의 픽셀이 가로로 나열되어 있는 형태이다.
num_train = x_train.shape[0]
num_test = x_test.shape[0]
np.random.seed(24)
NN = cm.NeuralNetwork()
NN.init_network()



def train():
    global NN
    batch_size = 30
    learning_rate = 0.05
    best_test_acc = 0
    best_params = None
    patience = 3 # 개선이 없을 때 최대 몇 번까지 기다릴지
    no_improve = 0 # 개선이 없는 에포크 수
    sample = 5000

    train_start = time.time()
    validation_data_set_amount = 5000
    x_train_learn = x_train[:55000]  # 학습용
    x_val = x_train[55000:]           # 검증용 (학습에 **미포함**)
    t_train_learn = t_train[:55000]
    t_val = t_train[55000:]

    for epoch in range(10):

        NN.layers['Dropout'].training_flag = True # 학습할 때는 Dropout 켜두기

        epoch_start = time.time()
        # Learning rate decay
        current_lr = learning_rate
        if epoch >= 5:
            current_lr = learning_rate * 0.5

        idx = np.random.permutation(len(x_train_learn))
        x_train_learn = x_train_learn[idx]
        t_train_learn = t_train_learn[idx]
        for i in range(0, num_train - validation_data_set_amount, batch_size):
            xb = x_train_learn[i:i+batch_size]
            tb = t_train_learn[i:i+batch_size]
            NN.gradient(xb, tb) # forward, backward

            # update
            for layer in NN.layers.values():
                if hasattr(layer, 'W'):
                    layer.W -= current_lr * (layer.dW + 0.0001 * layer.W)  # Weight decay 추가
                    layer.b -= current_lr * layer.db
        batch_time = time.time() - epoch_start

        NN.layers['Dropout'].training_flag = False # 평가할 때는 Dropout 안되도록 방지

        # --- 평가 코드 --- 

        # --- 훈련 성능 (앞에서 1000장만 샘플) ---
        eval_start = time.time()
        train_out = NN.predict(x_train_learn[:sample])
        train_out = af.softmax(train_out)
        train_loss = af.cross_entropy_error(train_out, t_train_learn[:sample])
        train_acc  = np.mean(np.argmax(train_out, axis=1) == np.argmax(t_train_learn[:sample], axis=1))

        # --- 테스트 성능 ---
        test_out = NN.predict(x_test[:sample])
        test_out = af.softmax(test_out)
        test_loss = af.cross_entropy_error(test_out, t_test[:sample])
        test_acc  = np.mean(np.argmax(test_out, axis=1) == np.argmax(t_test[:sample], axis=1))
        eval_time = time.time() - eval_start
        epoch_time = time.time() - epoch_start

        val_out = NN.predict(x_val[:sample])  # Validation Data Set 사용
        val_out = af.softmax(val_out)
        val_loss = af.cross_entropy_error(val_out, t_val[:sample])
        val_acc = np.mean(np.argmax(val_out, axis=1) == np.argmax(t_val[:sample], axis=1))
        print(f"Epoch {epoch}: train loss {train_loss:.4f} acc {train_acc:.3f} | test loss {test_loss:.4f} acc {test_acc:.3f} | val acc {val_acc:.3f}")
        print(f"time {epoch_time:.1f}s (batch {batch_time:.1f}s, eval {eval_time:.1f}s)")

        # --- 조기 종료 로직 ---
        if val_acc > best_test_acc:
            best_test_acc = val_acc
            best_params = [(layer.W.copy(), layer.b.copy()) if hasattr(layer, 'W') else None for layer in NN.layers.values()]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"조기 종료 (epoch {epoch}, 최고 test acc {best_test_acc:.3f})")
                break
    
    # 가장 좋았던 가중치로 복원
    if best_params is not None:
        for layer, snap in zip(NN.layers.values(), best_params):
            if snap is not None:
                layer.W, layer.b = snap
    
    total_time = time.time() - train_start
    print(f"훈련 완료 (총 {total_time:.1f}초, {total_time/60:.1f}분)")

def test():
    NN.layers['Dropout'].training_flag = False # 평가할 때는 Dropout 안되도록 방지
    output = NN.predict(x_test)
    pred = np.argmax(output, axis=1)
    true = np.argmax(t_test, axis=1)
    correct = np.sum(pred == true)
    print("정답률: " + str(correct / num_test * 100) + "%")

train()
test()