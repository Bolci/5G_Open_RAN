import numpy as np

# Example array of shape (72, 50)
array = np.arange(72*50).reshape((72,50))
print(array)

# Reshaping the array to (72, 2, 25)
reshaped_array = array.reshape(72, 25, 2)
result = reshaped_array.transpose(1, 0, 2)

print(result)
print(result.shape)