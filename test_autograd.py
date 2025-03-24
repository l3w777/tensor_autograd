import unittest
import numpy as np
from tensor import Tensor

class TestAutograd(unittest.TestCase):

    def test_add_backward(self):
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(3.0, requires_grad=True)
        c = a + b
        c.backward()
        self.assertEqual(a._grad, 1.0)
        self.assertEqual(b._grad, 1.0)

    def test_mul_backward(self):
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(3.0, requires_grad=True)
        c = a * b
        c.backward()
        self.assertEqual(a._grad, 3.0)
        self.assertEqual(b._grad, 2.0)

    def test_pow_backward(self):
        a = Tensor(2.0, requires_grad=True)
        b = 3
        c = a ** b
        c.backward()
        self.assertEqual(a._grad, 12.0)

    def test_relu_backward(self):
        a = Tensor(-2.0, requires_grad=True)
        b = a.relu()
        b.backward()
        self.assertEqual(a._grad, 0)

        a = Tensor(2.0, requires_grad=True)
        b = a.relu()
        b.backward()
        self.assertEqual(a._grad, 1)

    def test_complex_expression(self):
        # Пример сложного выражения для тестирования
        a = Tensor(1.0, requires_grad=True)
        b = Tensor(2.0, requires_grad=True)
        c = a * b + a**2
        c.backward()
        self.assertEqual(a._grad, 4.0)
        self.assertEqual(b._grad, 1.0)

    def test_neg_backward(self):  # Тест для отрицания
        a = Tensor(2.0, requires_grad=True)
        b = -a
        b.backward()
        self.assertEqual(a._grad, -1.0)

    def test_sub_backward(self):  # Тест для вычитания
        a = Tensor(5.0, requires_grad=True)
        b = Tensor(2.0, requires_grad=True)
        c = a - b
        c.backward()
        self.assertEqual(a._grad, 1.0)
        self.assertEqual(b._grad, -1.0)

    def test_rmul_backward(self):  # Тест для обратного умножения (other * self)
        a = Tensor(2.0, requires_grad=True)
        b = 3.0  # Обычное число (не Tensor)
        c = b * a  # b * a
        c.backward()
        self.assertEqual(a._grad, 3.0)

    def test_radd_backward(self):  # Тест для обратного сложения (other + self)
        a = Tensor(2.0, requires_grad=True)
        b = 3.0  # Обычное число (не Tensor)
        c = b + a  # b + a
        c.backward()
        self.assertEqual(a._grad, 1.0)

    def test_rsub_backward(self):  # Тест для обратного вычитания (other - self)
        a = Tensor(2.0, requires_grad=True)
        b = 5.0  # Обычное число (не Tensor)
        c = b - a  # b - a
        c.backward()
        self.assertEqual(a._grad, -1.0)

    def test_rtruediv_backward(self):  # Test for reverse true division (other / self)
        a = Tensor(2.0, requires_grad=True)
        b = 4.0
        c = b / a
        c.backward()
        self.assertEqual(a._grad, -1.0)

    def test_truediv_backward(self):  # Test for true division (self / other)
        a = Tensor(4.0, requires_grad=True)
        b = Tensor(2.0, requires_grad=True)
        c = a / b
        c.backward()
        self.assertEqual(a._grad, 0.5)
        self.assertEqual(b._grad, -1.0)


if __name__ == '__main__':
    unittest.main()
