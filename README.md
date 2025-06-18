# 📈 Regressão Linear Simples em Python (Manual)

Este repositório contém uma implementação **manual** de uma Regressão Linear Simples usando apenas **NumPy**, sem bibliotecas especializadas como `scikit-learn`.

O objetivo é aprender os fundamentos da regressão linear, entendendo cada passo do cálculo: coeficiente de correlação, inclinação da reta e intercepto.

---

## 🔍 O que este código faz:

- Calcula a **reta de regressão linear** para um conjunto de dados `x` e `y`
- Utiliza fórmulas estatísticas para:
  - Correlação
  - Inclinação da reta (coeficiente angular)
  - Intercepto (coeficiente linear)
- Permite prever valores de `y` com base em um novo valor de `x`

---

## 📌 Exemplo de uso:

```python
from numpy import *

class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.__correlation_coefficient = self.__correlacao()
        self.__inclination = self.__inclinacao()
        self.__intercept = self.__interceptacao()

    def __correlacao(self):
        covariacao = cov(self.x, self.y, bias=True)[0][1]
        variancia_x = var(self.x)
        variancia_y = var(self.y)
        return covariacao / sqrt(variancia_x * variancia_y) 
    
    def __inclinacao(self):
        stdx = std(self.x)
        stdy = std(self.y)
        return self.__correlation_coefficient * (stdy / stdx)

    def __interceptacao(self):
        mediax = mean(self.x)
        mediay = mean(self.y)
        return mediay - self.__inclination * mediax

    def previsao(self, valor):
        return self.__intercept + (self.__inclination * valor)

# Exemplo
x = array([1, 2, 3, 4, 5])
y = array([2, 4, 6, 8, 10])

lr = LinearRegression(x, y)
print("Previsão para x=6:", lr.previsao(6))
