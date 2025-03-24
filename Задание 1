import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self._grad = 0 if requires_grad else None  # Инициализируем градиент нулем, если requires_grad=True
        self._backward = lambda: None  # Функция для вычисления обратного прохода (по умолчанию ничего не делает)
        self._prev = set(_children)
        self._op = _op
        self.id = Tensor.generate_id()

    _next_id = 0  # Статическая переменная для генерации ID

    @staticmethod
    def generate_id():
        Tensor._next_id += 1
        return Tensor._next_id - 1

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                self._grad += out._grad * 1.0  # Умножаем на 1.0 для преобразования в float
            if other.requires_grad:
                other._grad += out._grad * 1.0 # Умножаем на 1.0 для преобразования в float
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                self._grad += out._grad * other.data
            if other.requires_grad:
                other._grad += out._grad * self.data
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, requires_grad=self.requires_grad, _children=(self,), _op=f'**{other}')

        def _backward():
            if self.requires_grad:
                self._grad += out._grad * other * (self.data**(other-1))
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad, _children=(self,), _op='ReLU')

        def _backward():
            if self.requires_grad:
                self._grad += out._grad * (self.data > 0)
        out._backward = _backward

        return out

    def backward(self):
        # Topological sort of the graph
        topo = []
        visited = set()
        def topological_sort(v):
            visited.add(v)
            for child in v._prev:
                if child not in visited:
                    topological_sort(child)
            topo.append(v)
        topological_sort(self)

        self._grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self._grad if self.requires_grad else None}, id={self.id})"
