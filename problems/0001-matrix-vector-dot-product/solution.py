def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	if len(a[0]) != len(b):
		return  -1 
	result = []
	for i in a: 
		aux = 0
		for j in range(0,len(a)): 
			aux += i[j] * b[j]
		result.append(aux)
	return  result
