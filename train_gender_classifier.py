import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка данных (замените 'gender_data.csv' на ваш файл)
data = pd.read_csv('gender_data.csv')

# Преобразование данных в numpy массивы
X = data[['height', 'weight']].values  # Признаки: рост и вес
y = data['gender'].values.reshape(-1, 1) # Целевая переменная: пол (0 или 1), преобразуем в вектор-столбец

# Нормализация данных (важно для улучшения сходимости)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Разбиение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================================================
# Определение классов Tensor, SGD, Layer, Linear, Sequential, Sigmoid, BCELoss
# ==========================================================================

class Tensor(object):
    grad = None  # Градиент тензора
    def __init__(self, data, creators = None, operation_on_creation = None,  autograd=False, id=None):
        self.data = np.array(data)  # Данные тензора (NumPy массив)
        self.creators = creators     # Список тензоров, из которых был создан этот тензор
        self.operation_on_creation = operation_on_creation # Операция, использованная для создания тензора
        self.autograd = autograd       # Включен ли автоматический дифференцировщик
        self.children = {}           # Словарь, отслеживающий, сколько раз этот тензор использовался при создании других

        if id is None:
            self.id = np.random.randint(0, 1000000000)  # Уникальный ID для отслеживания

        if (self.creators is not None):
            for creator in creators:  # Увеличиваем счетчик использования для каждого создателя
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] +=1


    def __add__(self, other):  # Перегрузка оператора +
        if self.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        else:
            return Tensor(self.data + other.data)

    def __str__(self): # Преобразование в строку
        return str(self.data)

    def backward(self, grad=None, grad_child=None):  # Обратное распространение ошибки
        if self.autograd:  # Если требуется автоматическое дифференцирование
            if grad is None:
                grad = Tensor(np.ones_like(self.data))  # Если градиент не задан, начинаем с единиц
            if grad_child is not None:
                if (self.children[grad_child.id]) > 0:
                    self.children[grad_child.id] -=1 # Уменьшаем счетчик использования дочерним элементом
            if self.grad is None:
                self.grad = grad  # Записываем градиент
            else:
                self.grad += grad # Накапливаем градиент

            if (self.creators is not None and (self.check_grads_from_child() or grad_child is None)): # Если есть создатели и все градиенты от дочерних получены
                if (self.operation_on_creation == "+"):
                    self.creators[0].backward(self.grad, self)  # Передаем градиент создателям
                    self.creators[1].backward(self.grad,self)
                elif (self.operation_on_creation == "-1"):
                    self.creators[0].backward(self.grad.__neg__(), self)
                elif (self.operation_on_creation == "-"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad.__neg__(),self)
                elif "sum" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(self.grad.expand(axis, self.creators[0].data.shape[axis]), self)
                elif "expand" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(self.grad.sum(axis), self)
                elif (self.operation_on_creation == "*"):
                    self.creators[0].backward(self.grad * self.creators[1], self)
                    self.creators[1].backward(self.grad * self.creators[0], self)
                elif "dot" in self.operation_on_creation:
                    temp = self.grad.dot(self.creators[1].transpose())
                    self.creators[0].backward(temp, self)
                    temp = self.grad.transpose().dot(self.creators[0]).transpose()
                    self.creators[1].backward(temp,self)
                elif "transpose" in self.operation_on_creation:
                    self.creators[0].backward(self.grad.transpose(), self)
                elif "sigmoid" in self.operation_on_creation:
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad*(self * (temp - self)), self) # Градиент сигмоиды
                elif "tanh" in self.operation_on_creation:
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad*(temp - (self * self)), self) # Градиент tanh
                elif "softmax" in self.operation_on_creation:
                    self.creators[0].backward(Tensor(self.grad.data), self)

    def __neg__(self):  # Перегрузка оператора - (унарный минус)
        if self.autograd:
            return Tensor(self.data * -1, [self], "-1", True)
        else:
            return Tensor(self.data * -1)

    def __sub__(self, other): # Перегрузка оператора -
        if self.autograd:
            return Tensor(self.data - other.data, [self, other], "-", True)
        else:
            return Tensor(self.data - other.data)

    def __mul__(self, other): # Перегрузка оператора *
        if self.autograd:
            return Tensor(self.data * other.data, [self, other], "*", True)
        else:
            return Tensor(self.data * other.data)

    def sum(self, axis):  # Сумма по оси
        if self.autograd:
            return Tensor(self.data.sum(axis), [self], "sum_"+str(axis),True)
        return Tensor(self.data.sum(axis))

    def expand(self, axis, count_copies): # Расширение размерности
        transpose = list(range(0, len(self.data.shape)))
        transpose.insert(axis, len(self.data.shape))
        expand_shape= list(self.data.shape) + [count_copies]
        expand_data = self.data.repeat(count_copies).reshape(expand_shape)
        expand_data = expand_data.transpose(transpose)
        if (self.autograd):
            return Tensor(expand_data, [self], "expand_"+str(axis), autograd=True)
        return Tensor(expand_data)

    def dot(self, other):  # Матричное умножение
        if self.autograd:
            return Tensor(self.data.dot(other.data), [self, other], "dot", True)
        else:
            return Tensor(self.data.dot(other.data))

    def transpose(self):   # Транспонирование
        if self.autograd:
            return Tensor(self.data.transpose(), [self], "transpose", True)
        else:
            return Tensor(self.data.transpose())

    def sigmoid(self):     # Сигмоидальная функция
        if self.autograd:
            return Tensor(1/(1+np.exp(-self.data)), [self], "sigmoid", True)
        else:
            return Tensor(1/(1+np.exp(-self.data)))

    def tanh(self):       # Гиперболический тангенс
        if self.autograd:
            return Tensor(np.tanh(self.data), [self], "tanh", True)
        else:
            return Tensor(np.tanh(self.data))

    def softmax(self):    # Softmax функция
        exp = np.exp(self.data)
        exp = exp/np.sum(exp, axis = 1, keepdims=True)
        if self.autograd:
            return Tensor(exp, [self], "softmax", True)
        return Tensor(exp)

    def check_grads_from_child(self): # Проверка, получены ли все градиенты от дочерних элементов
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True

    def __repr__(self):  # Представление объекта
        return str(self.data.__repr__())

    def log(self): #  Натуральный логарифм
        if self.autograd:
            return Tensor(np.log(self.data), [self], "log", True)
        return Tensor(np.log(self.data))

class SGD(object):  # Стохастический градиентный спуск
    def __init__(self, weigts, learning_rate):
        self.weights = weigts       # Веса для обновления
        self.learning_rate = learning_rate  # Скорость обучения

    def step(self):  # Шаг обновления весов
        for weight in self.weights:
            weight.data -=self.learning_rate * weight.grad.data  # Обновляем веса
            weight.grad.data *= 0 # Обнуляем градиенты

class Layer(object):  # Базовый класс для слоев
    def __init__(self):
        self.parameters = [] # Список параметров слоя

    def get_parameters(self):
        return self.parameters # Возвращает список параметров

class Linear(Layer):  # Полносвязный (линейный) слой
    def __init__(self, input_count, output_count):
        super().__init__()
        weight = np.random.randn(input_count, output_count) * np.sqrt(2.0/input_count) # Инициализация весов (He initialization)
        self.weight = Tensor(weight,autograd=True) # Веса как Tensor
        self.bias = Tensor(np.zeros(output_count), autograd=True) # Смещения как Tensor
        self.parameters.append(self.weight) # Добавляем вес в список параметров
        self.parameters.append(self.bias)  # Добавляем смещение в список параметров

    def forward(self, inp):  # Прямой проход
        return inp.dot(self.weight) + self.bias.expand(0,len(inp.data)) # Линейная комбинация:  X * W + b

class Sequential(Layer): # Контейнер для последовательного выполнения слоев
    def __init__(self, layers):
        super().__init__()
        self.layers = layers # Список слоев

    def add(self, layer):
        self.layers.append(layer) # Добавляет слой в последовательность

    def forward(self, inp): # Прямой проход через все слои
        for layer in self.layers:
            inp = layer.forward(inp) # Выполняем прямой проход для каждого слоя
        return inp

    def get_parameters(self): #  Возвращает все параметры всех слоев
        params = []
        for layer in self.layers:
            params += layer.get_parameters()
        return params

class Sigmoid(Layer): # Слой сигмоиды
    def forward(self, inp):
        return inp.sigmoid() # Применяет сигмоиду к входу

class Tanh(Layer):   # Слой гиперболического тангенса
    def forward(self, inp):
        return inp.tanh() # Применяет tanh к входу

class Softmax(Layer): # Слой Softmax
    def forward(self, inp):
        return inp.softmax() # Применяет softmax к входу

class BCELoss(Layer): # Loss функция BinaryCrossEntropy
    def forward(self, prediction, target):
        # Вычисление Binary Cross Entropy loss
        return -((target * prediction.log()) + ((1-target)*( (1-prediction).log()))).sum(0)


# ==========================================================================
# Обучение модели
# ==========================================================================

np.random.seed(0) # фиксируем случайное зерно для воспроизводимости результатов

num_epoch = 1000  # Количество эпох обучения

# Model (определяем архитектуру сети)
model = Sequential([
    Linear(2, 4),   # 2 входа (рост, вес), 4 нейрона в скрытом слое
    Sigmoid(),      # Сигмоидальная активация в скрытом слое
    Linear(4, 1),   # 4 входа, 1 выход (вероятность пола)
    Sigmoid()       # Сигмоидальная активация на выходе
])

# Optimizer (определяем оптимизатор)
sgd = SGD(model.get_parameters(), 0.01)

# Loss Function (определяем функцию потерь)
loss = BCELoss()

# Training Loop (цикл обучения)
for i in range(num_epoch):
    # Create Tensors (создаем Tensor из numpy массивов)
    X_train_tensor = Tensor(X_train, autograd = True) # Входные данные для обучения
    y_train_tensor = Tensor(y_train, autograd = True) # Целевые данные для обучения

    # Model Prediction (вычисляем предсказания модели)
    y_hat = model.forward(X_train_tensor) # Прямой проход

    # Calculate Loss (вычисляем функцию потерь)
    error = loss.forward(y_hat, y_train_tensor) # Вычисляем ошибку между предсказанием и целью

    # Pass the Gradients (передаем градиенты)
    error.backward(Tensor(np.ones_like(error.data)))  # Обратное распространение

    # Update the weights (обновляем веса модели)
    sgd.step() # Шаг оптимизации

    #Print Loss (выводим значение функции потерь для отслеживания прогресса)
    if i % 50 == 0:
        print(f"Loss: {error}")

# ==========================================================================
# Оценка модели
# ==========================================================================

# Evaluation (Make Prediction) (функция для предсказания класса)
def predict(input):
    output_layer = model.forward(input) # Прямой проход
    if output_layer.data[0][0] > 0.5: # Если вероятность > 0.5, то предсказываем класс 1 (мужской)
        return 1 # Male
    else:
        return 0 # Female

# Testing data
X_test_tensor = Tensor(X_test, autograd = False) # Создаем Tensor из тестовых данных

y_test_pred = [] # Список для хранения предсказанных классов

# Inference (цикл для предсказания классов для тестовых данных)
for i in range(len(X_test)):
    test_sample = Tensor(X_test[i].reshape(1, 2), autograd = False)  # Создаем Tensor для одного тестового примера
    prediction = predict(test_sample) # Предсказываем класс
    y_test_pred.append(prediction) # Добавляем предсказание в список

# Evaluation (Calculate Accuracy) (вычисляем точность)
correct = 0
for i in range(len(y_test)):
    if y_test_pred[i] == y_test[i][0]: # Если предсказанный класс совпадает с фактическим классом
        correct += 1 # Увеличиваем счетчик правильных предсказаний

print(f"Accuracy: {correct/len(y_test)}") # Выводим точность модели
