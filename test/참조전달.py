x = [1, 2, 3] # Immutable
y = 3 # Mutable

def f():
    global x
    global y
    x[1] += 1
    y += 4

def g(f, x, y):
    x[1] += 3
    y += 6
    f()
    
g(f, x, y)
print(x, y)

'''
불변 객체 (Immutable) | int, float, str, tuple      | 값을 바꿀 수 없으므로, 새로운 객체가 생성됨 (원본 영향 없음)
가변 객체 (Mutable)	  | numpy.ndarray, list, dict	| 내부 원소를 직접 수정하면, 원본 객체가 그대로 수정됨
'''