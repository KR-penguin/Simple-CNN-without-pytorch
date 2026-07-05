import numpy as np
import activation_function as af
import my_cnn_module as cm

x_train, x_test, t_train, t_test = cm.get_test_data()
# 데이터 행렬 크기 = (60000, 784) = (훈련 데이터의 수, 이미지의 픽셀 수)
# 각 데이터의 픽셀이 가로로 나열되어 있는 형태이다.
num_train = x_train.shape[0]
num_test = x_test.shape[0]
np.random.seed(42)
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

    for epoch in range(10):
        for i in range(0, num_train, batch_size):
            xb = x_train[i:i+batch_size] 
            tb = t_train[i:i+batch_size] 
            NN.gradient(xb, tb) # forward, backward

            # update
            for layer in NN.layers.values():
                if hasattr(layer, 'W'):
                    layer.W -= learning_rate * layer.dW
                    layer.b -= learning_rate * layer.db

        # --- 훈련 성능 (앞에서 1000장만 샘플) ---
        train_out = NN.predict(x_train[:sample])
        train_out = af.softmax(train_out)
        train_loss = af.cross_entropy_error(train_out, t_train[:sample])
        train_acc  = np.mean(np.argmax(train_out, axis=1) == np.argmax(t_train[:sample], axis=1))
        
        # --- 테스트 성능 ---
        test_out = NN.predict(x_test[:sample])
        test_out = af.softmax(test_out)
        test_loss = af.cross_entropy_error(test_out, t_test[:sample])
        test_acc  = np.mean(np.argmax(test_out, axis=1) == np.argmax(t_test[:sample], axis=1))
        
        print(f"Epoch {epoch}: train loss {train_loss:.4f} acc {train_acc:.3f} | test loss {test_loss:.4f} acc {test_acc:.3f}")
    
        # --- 조기 종료 로직 ---
        if test_acc > best_test_acc:
            best_test_acc = test_acc
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
    
    print("훈련 완료")

def test():
    output = NN.predict(x_test)
    pred = np.argmax(output, axis=1)
    true = np.argmax(t_test, axis=1)
    correct = np.sum(pred == true)
    print("정답률: " + str(correct / num_test * 100) + "%")

train()
test()